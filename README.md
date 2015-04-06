## transformdata.py

* **Run:** ./transformdata.py _output_filename_ 
* transformdata.py builds a data set from sentences.xml and saves it in _output_filename_.

## train.py 

* **Run:** ./train.py _dataset_filename_ { _train_, _assess_ }
* train.py can does 2 things: 
    1. trains model and save it in _classifier.plk_ file;
    2. assesses model (cross validation)
    
## predict.py

* **Run:** ./predict.py _file_with_text_
* predict.py reads _file_with_text_ and splits it into sentences