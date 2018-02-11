#!/bin/bash

#python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_nonrotate_600p/models.ckpt-990 ../data/train_single_class_ae_plane_chamfer_nonrotate_600p/train_pc.pkl False  600
#python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_nonrotate_600p/models.ckpt-990 ../data/val_single_class_ae_plane_chamfer_nonrotate_600p/val_pc.pkl False  600
#python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_nonrotate_600p/models.ckpt-990 ../data/test_single_class_ae_plane_chamfer_nonrotate_600p/test_pc.pkl False  600

#python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_zrotate4dir_600p/models.ckpt-990 ../data/train_single_class_ae_plane_chamfer_zrotate4dir_600p/train_pc.pkl True  600
#python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_zrotate4dir_600p/models.ckpt-990 ../data/val_single_class_ae_plane_chamfer_zrotate4dir_600p/val_pc.pkl True  600
#python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_zrotate4dir_600p/models.ckpt-990 ../data/test_single_class_ae_plane_chamfer_zrotate4dir_600p/test_pc.pkl True  600

python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_nonrotate_2048p/models.ckpt-990 ../data/train_single_class_ae_plane_chamfer_nonrotate_2048p/train_pc.pkl False  2048
python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_nonrotate_2048p/models.ckpt-990 ../data/val_single_class_ae_plane_chamfer_nonrotate_2048p/val_pc.pkl False  2048
python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_nonrotate_2048p/models.ckpt-990 ../data/test_single_class_ae_plane_chamfer_nonrotate_2048p/test_pc.pkl False  2048

python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_zrotate4dir_2048p/models.ckpt-990 ../data/train_single_class_ae_plane_chamfer_zrotate4dir_2048p/train_pc.pkl True  2048
python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_zrotate4dir_2048p/models.ckpt-990 ../data/val_single_class_ae_plane_chamfer_zrotate4dir_2048p/val_pc.pkl True  2048
python generate_hidden_ae_model.py ../data/train_single_class_ae_plane_chamfer_zrotate4dir_2048p/models.ckpt-990 ../data/test_single_class_ae_plane_chamfer_zrotate4dir_2048p/test_pc.pkl True  2048
