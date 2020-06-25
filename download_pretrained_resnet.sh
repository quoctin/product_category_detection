#!/bin/bash

if [ ! -d "pretrained" ]
then
	mkdir "pretrained"
	wget "http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz"
	mv "resnet_v2_50_2017_04_14.tar.gz" "pretrained"
	cd "pretrained"
	mkdir "resnet_v2_50_2017_04_14"
	tar xopf "resnet_v2_50_2017_04_14.tar.gz" -C "resnet_v2_50_2017_04_14"
	rm "resnet_v2_50_2017_04_14.tar.gz"
	cd ..
	wget "http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz"
	mv "resnet_v2_101_2017_04_14.tar.gz" "pretrained"
	cd "pretrained"
	mkdir "resnet_v2_101_2017_04_14"
	tar xopf "resnet_v2_101_2017_04_14.tar.gz" -C "resnet_v2_101_2017_04_14"
	rm "resnet_v2_101_2017_04_14.tar.gz"
	cd ..
fi