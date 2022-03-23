echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running brevitasConverter.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 brevitasConverter.py --input=ExampleModels/mobilenet.py --output=mobilenetQuant.py --engine=Engine/quantizers.py --weightsEngine=CustomWeightQuant --activationsEngine=CustomActQuant --inputBitWidth=8
echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running blankGenerator.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 blankGenerator.py --modelFile=mobilenetQuant.py --model=MobileNetQuant --output=blank.pth
echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Running weightsConverter.py
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
python3 weightsConverter.py --original=TrainedModel/originalMobileNet.pth --blank=blank.pth --out=ckpt.pth
echo $'\n'=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
echo Tidying files into Output folder
echo =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/$'\n'
mkdir -p Output
rm blank.pth
mv mobilenetQuant.py Output/mobilenetQuant.py
mv ckpt.pth Output/ckpt.pth
