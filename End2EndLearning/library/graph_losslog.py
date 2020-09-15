import plotly.express as px
import pandas as pd 
import os

LOSS_LOG_FILE = "~/Downloads/loss-log"
OUTPUT_FOLDER = "./graphs/"

def graph_accuracy(filename, outputFilename=os.path.join(OUTPUT_FOLDER+"acc_graph.png"), save=False):
    df = pd.read_csv(LOSS_LOG_FILE)
    fig = px.line(df, x="epoch", y="accuracy")

    if save:
        fig.write_image(outputFilename)

    fig.show()

def graph_loss(filename, outputFilename=os.path.join(OUTPUT_FOLDER+"loss_graph.png"), save=False):
    df = pd.read_csv(LOSS_LOG_FILE)
    fig = px.line(df, x="epoch", y="loss")

    if save:
        fig.write_image(outputFilename)
        
    fig.show()

def graph_val_accuracy(filename, outputFilename=os.path.join(OUTPUT_FOLDER+"val_acc_graph.png"), save=False):
    df = pd.read_csv(LOSS_LOG_FILE)
    fig = px.line(df, x="epoch", y="val_accuracy")

    if save:
        fig.write_image(outputFilename)
        
    fig.show()

def graph_val_loss(filename, outputFilename=os.path.join(OUTPUT_FOLDER+"val_loss_graph.png"), save=False):
    df = pd.read_csv(LOSS_LOG_FILE)
    fig = px.line(df, x="epoch", y="val_loss")

    if save:
        fig.write_image(outputFilename)
        
    fig.show()

if __name__=="__main__":

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    graph_accuracy(LOSS_LOG_FILE, save=True)
    graph_loss(LOSS_LOG_FILE, save=True)
    graph_val_accuracy(LOSS_LOG_FILE,save=True)
    graph_val_loss(LOSS_LOG_FILE, save=True)