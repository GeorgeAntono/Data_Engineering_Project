gsutil cp ../data/* gs://data_degroup11

git config --global user.email "georgantonopoulos@hotmail.com"
git config --global user.name "GeorgeAntono"
echo "data uploaded" >> data/history.txt
git commit -am "data uploaded"
git push https://$1:$2@github.com/GeorgeAntono/DE_Group11.git --all

