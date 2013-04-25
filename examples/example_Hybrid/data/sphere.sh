rtkdrawgeometricphantom \
    --phantomfile sphere.txt \
    --dimension 256 \
    --spacing 1\
    --output sphere.mha
clitkImageConvert -i sphere.mha -o sphere.mha -c

