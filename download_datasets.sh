wget https://polybox.ethz.ch/index.php/s/6cYYSheXDP6ZiC5/download -O datasets/ptb_xl.zip
unzip datasets/ptb_xl.zip -d datasets/ptb_xl
touch datasets/ptb_xl/download_timestamp.txt
rm datasets/ptb_xl.zip

wget https://polybox.ethz.ch/index.php/s/7hE6WIct12CZi66/download -O datasets/backgrounds.zip
mkdir datasets/backgrounds
unzip datasets/backgrounds.zip -d datasets/backgrounds
touch datasets/backgrounds/download_timestamp.txt
rm datasets/backgrounds.zip

mkdir downloaded_checkpoints
wget https://polybox.ethz.ch/index.php/s/Sf4LtNsbAJmzt0t/download -O downloaded_checkpoints/segmentation_model.zip
unzip downloaded_checkpoints/segmentation_model.zip -d downloaded_checkpoints/
rm downloaded_checkpoints/segmentation_model.zip