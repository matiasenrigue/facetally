create_folders:
	@mkdir ~/.lewagon/facetally_data
	@mkdir ~/.lewagon/facetally_data/image_data
	@echo Successfully created folders

copy_from_old_structure:
	@cp raw_data/bbox_train.csv ~/.lewagon/facetally_data/bbox_train.csv
	@cp -r raw_data/image_data ~/.lewagon/facetally_data/

#################### PACKAGE ACTIONS ###################
install:
	@pip install -e .

reinstall:
	@pip uninstall -y facetally || :
	@pip install -e .

update_raw_data_from_GCP:
	python -c 'from face_tally.ml_logic.data import update_local_raw_data_from_GCP; update_local_raw_data_from_GCP()'

test_preproc:
	@python face_tally/visulization/visualisation.py

test_training:
	@python face_tally/visulization/visualisation.py