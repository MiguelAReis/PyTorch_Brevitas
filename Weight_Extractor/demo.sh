echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running extractWeights.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 extractWeights.py --pth=final9090_9276.pth --modelFile=deeplabQuantBNMerged.py --model=DeepLabQuant --modelParams "num_classes=2,output_stride=16" --weights=4 --activations=4
