echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running brevitasConverter.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 brevitasConverter.py --input=deeplab.py --output=deeplabQuant.py --engine=Engine/quantizers.py --weightsEngine=CustomWeightQuant --activationsEngine=CustomActQuant --inputBitWidth=8
echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running blankGenerator.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 blankGenerator.py --modelFile=deeplabQuant.py --model=DeepLabQuant --output=blank.pth
echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running weightsConverter.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 weightsConverter.py --original=deepLab_92,45%.pth --blank=blank.pth --out=ckpt.pth
echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Tidying files into Output folder
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
rm blank.pth
