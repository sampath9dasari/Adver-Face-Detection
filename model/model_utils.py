#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:53:39 2020

@author: susanthdasari
"""
import torch
import os
from pathlib import Path

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def save_checkpoint(state, filename="checkpoint.pth", save_path="saved_models"):
    # check if the save directory exists
    if not Path(os.getcwd()+'/'+save_path).exists():
        Path(os.getcwd()+'/'+save_path).mkdir()

    save_path = Path(os.getcwd()+'/'+save_path, filename)
    torch.save(state, str(save_path))


def save_losses(state, filename="checkpoint.pth", save_path="saved_models"):
    # check if the save directory exists
    if not Path(os.getcwd()+'/'+save_path).exists():
        Path(os.getcwd()+'/'+save_path).mkdir()

    save_path = Path(os.getcwd()+'/'+save_path, filename)
    torch.save(state, str(save_path))