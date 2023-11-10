import torch
import argparse
import torchvision
from torch import nn, optim
from model_training import train
from torchvision import transforms
from timeit import default_timer as timer
from utils import plot_loss_curves, save_model
from data_preprocess import create_dataloaders
from models import create_effinetb2_model, create_vit_model, TinyVGG

torch.manual_seed(1126)
torch.cuda.manual_seed(1126)
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--m', dest='model_type', default='tinyvgg', help='set model for training (default: vit)')
    args.add_argument('--e', dest='epochs', default=20, type=int, help='training epochs')
    args = args.parse_args()
    
    epochs = args.epochs
    print(f"Device:{device}, model:{args.model_type}, training epochs: {epochs}")

    ## get model & transformer & data loader
    if args.model_type == 'vit':
        model, transformer = create_vit_model()
        train_transformer = transformer
        
    elif args.model_type =='effnet':
        model, transformer = create_effinetb2_model()
        train_transformer = transforms.Compose([transforms.TrivialAugmentWide(),
                                                transformer])

    elif args.model_type == 'tinyvgg':
        model = TinyVGG(input_shape=3, hidden_units=10)
        train_transformer = transforms.Compose([transforms.Resize((64, 64)),
                                                transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                                transforms.ToTensor()])

        transformer = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor()])

    ## create data loader
    train_dataloader, test_dataloader, class_names = create_dataloaders(batch_size  = 32,
                                                                        split_size = 0.2,
                                                                        train_transform = train_transformer,
                                                                        test_transform = transformer)
    ## create loss func & optimizer
    print("=== create loss func & optimizer ===")
    loss_fn = nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters())

    ## model training
    print("=== Model training ===")
    start_time = timer()
    model_results = train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn = loss_fn,
                            epochs=epochs)
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")


    ## print model performance
    print("=== models performance ===")
    for k, v in model_results.items():
        print(f"{k}: {v[-1]:.3f}")

    ## plot model result
    print("=== plot model loss & accuracy curve ===")
    plot_loss_curves(f"{args.model_type.capitalize()} Result" ,model_results)

    ## model to file
    print("=== model to file ===")
    save_model(model=model, target_dir="models", model_name=f"{args.model_type}_clf.pth")