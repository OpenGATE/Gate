rtkdrawgeometricphantom \
    --phantomfile cylinder.txt \
    --dimension 256 \
    --spacing 1\
    --output cylinder.mha
clitkImageConvert -i cylinder.mha -o cylinder.mha -c

