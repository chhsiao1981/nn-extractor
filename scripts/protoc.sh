#!/bin/bash

protoc --proto_path=./proto --python_out=./nn_extractor proto/nnextractor.proto
