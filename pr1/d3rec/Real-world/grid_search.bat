:: Deneme 1: Standart (Senin yaptığın)
python train.py --dataset_name ml-1m --drop_num 5 --lamda 1 --w_max 1.6 --epochs 200

:: Deneme 2: Daha güçlü kategori ayrımı (lamda artırıldı)
python train.py --dataset_name ml-1m --drop_num 5 --lamda 5 --w_max 1.6 --epochs 200

:: Deneme 3: Daha zayıf ağırlıklandırma (w_max düşürüldü)
python train.py --dataset_name ml-1m --drop_num 5 --lamda 1 --w_max 1.2 --epochs 200