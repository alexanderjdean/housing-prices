echo "Starting now. Downloading the data set..."
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip
rm -r data_description.txt
rm -r sample_submission.csv
rm -r house-prices-advanced-regression-techniques.zip
mkdir data/
mv train.csv data/
mv test.csv data/

echo "Processing the data set now..."
python src/preprocess.py
rm -r data/train.csv
rm -r data/test.csv

echo "Done processing! :)"