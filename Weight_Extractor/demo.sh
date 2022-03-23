echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running extractWeights.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 extractWeights.py --pth=ckpt.pth --modelFile=lenetQuant.py --model=LeNetQuant --weights=8 --activations=8
