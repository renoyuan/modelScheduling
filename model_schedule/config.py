#!/usr/bin/env python
#-*- coding: utf-8 -*-
#PROJECT_NAME: /home/reno/tlcvinvoicedetection/invoice_detection
#CREATE_TIME: 2023-10-18 
#E_MAIL: renoyuan@foxmail.com
#AUTHOR: reno 
#note:  InvoiceDetection 全局配置信息


import argparse

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(description='description')
    parser.add_argument("--name", type=str, default="name")
    parser.add_argument("--version", type=str, default="0.0.1")
    parser.add_argument("--schedule_mode", type=str, default="single",help=" single |  mul_thread  | async  ")
    parser.add_argument("--predictor_num", type=int, default=1)

    # params for model
    parser.add_argument("--model_version", type=str, default="0.0.1")
    parser.add_argument("--dl_framework", type=str, default='torch',help="深度学习框架")
    parser.add_argument("--model_path", type=str, default="",help="模型路径")
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu_id", type=int, default=0)    
    return parser.parse_args()

    

            