import os
import kagglehub

def data_collect(link):
    save_dir='data'
    os.makedirs(save_dir,exist_ok=True)
    path = kagglehub.dataset_download(link,path=save_dir)
    print("Path to dataset files:", path)

    return 0

def main():
    data_collect(link='"aksha05/flower-image-dataset"')

if __name__ == "__main__":
    main()