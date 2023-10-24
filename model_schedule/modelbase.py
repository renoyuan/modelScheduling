#!/usr/bin/env python
#-*- coding: utf-8 -*-
#PROJECT_NAME: /home/reno/tlcvtextdetection/text_detection
#CREATE_TIME: 2023-10-23 
#E_MAIL: renoyuan@foxmail.com
#AUTHOR: reno 
#note:  模型调度基础类

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
from multiprocessing import Process, Value
import asyncio
import threading
from abc import ABC, abstractmethod
from loguru import logger

try:
    import paddle
except:
    logger.warning(f"import paddle faild")

try:
    import torch
except:
    logger.warning(f"import torch faild")

class ModelScheduling(object):
    """模型调度"""
    
    def __init__(self,args,**kwargs ):
        """
        predictor_num: 模型调度器数量
        mode 模型调度模式  single |  mul_thread  | async  
        """
        self.args = args
        self.schedule_mode = self.args.schedule_mode
        self.predictor_num = self.args.predictor_num

        # 创建模型predictor
        self.predictors = []
        
        for i in range(self.predictor_num):
            predictor = self.create_predictor(**kwargs)
            logger.info(f"加载模型实例{predictor}")
            self.predictors.append({
                "predictor":predictor,
            })
            if self.schedule_mode in ("mul_thread"):
                self.predictors[i]["lock"] = threading.Lock() # 加锁全局锁,
            if self.schedule_mode in ("mul_process"):
                self.pool = multiprocessing.Pool(processes = 3)

            
    @abstractmethod
    def create_predictor(self,**kwargs):
        """创建调度器子类中实现"""
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def predict(self,):
        """调用模型pipelie子类中实现"""
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    async def predict_by_async(self,):
        """调用模型pipelie子类中实现"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def wrap_predict_by_thread(self,*args,**kwargs):
        """
        args:
        date, data_idx, predictor_idx
        """

        logger.info(f"excuteing predictor_idx:{args[2]} date_idx {args[1]}")
        lock = self.predictors[args[2]]["lock"]
        with lock:
            return self.predict(*args,**kwargs)

    def multy_thread_predict(self, task_date:list):
        """多线程调度"""
        predictor_idx = 0
        res_map = {}
        future_tasks = []
        with ThreadPoolExecutor(max_workers = self.predictor_num) as executor:
            logger.info(f"multy_thread_predict")
            for data_idx, date in enumerate(task_date):
                if predictor_idx>=self.predictor_num:
                    predictor_idx = 0
                logger.info(f"predictor_idx{predictor_idx}")
                future_tasks.append(executor.submit(self.wrap_predict_by_thread, date, data_idx, predictor_idx)) 
                predictor_idx +=1 
            
            # 等所有任务完毕
            wait(future_tasks, return_when=FIRST_COMPLETED)

            for future in future_tasks:
                res,data_idx = future.result()
                res_map[data_idx] = res
                
        return res_map
    
    async def async_task_run(self,task_date:list):
        """携程调度"""
        tasks = []
        res_map = {}
        logger.info(f"async_thread_predict")
        predictor_idx = 0
        for task_idx, date in enumerate(task_date):
            if predictor_idx>=self.predictor_num:
                    predictor_idx = 0
            tasks.append(self.predict_by_async(date, task_idx, predictor_idx))
            predictor_idx +=1
        results = await asyncio.gather(*tasks)
        for res,task_idx in results:
            res_map[task_idx] = res
        return res_map
    
    def multy_process_predict(self, data_list:list):
        """多进程调度"""
        res_map = {}
        result = []
        predictor_idx = 0
        for data_idx,data in enumerate(data_list):
            if predictor_idx>=self.predictor_num:
                    predictor_idx = 0
            result.append (self.pool.apply_async(self.predict, (data,data_idx, predictor_idx )))
            predictor_idx +=1 
        self.pool.close()
        self.pool.join()
        for future in result:
            res,data_idx = future.get()
            res_map[data_idx] = res
        return res_map
    
    def __call__(self, data_list:list):
        
        assert (isinstance(data_list,list) and len(data_list) >0),"Parameter Error "

        dt_boxes_map = {}
        if self.schedule_mode == "single" :
            for data_idx, data in enumerate(data_list):
                res,data_idx = self.predict(data,data_idx,0)
                dt_boxes_map[data_idx] = res
            return dt_boxes_map
        elif self.schedule_mode == "mul_thread" :
            dt_boxes_map = self.multy_thread_predict(data_list)
            return dt_boxes_map
        elif self.schedule_mode == "async":
            dt_boxes_map = asyncio.run(self.async_task_run(data_list))
            return dt_boxes_map
        else:
            return self.multy_process_predict(data_list) 
        
    def __del__(self):
        
        logger.info("销毁实例：{}".format(str(self)))
        for i in self.predictors:
            if self.args.dl_framework == "paddle":
                i["predictor"].clear_intermediate_tensor()
                i["predictor"].try_shrink_memory()
            del i["predictor"]
        
            
        if self.args.dl_framework == "paddle":
         
            paddle.disable_static()
            paddle.device.cuda.empty_cache()
            
        elif self.args.dl_framework == "torch":
        
            torch.cuda.empty_cache()
            
       
        
        del self
        
    def disposal(self):
       
        self.__del__()