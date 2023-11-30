install:
	@pip install -e .

test_preproc:
	@python face_tally/visulization/visualisation.py

create_folders:
	@mkdir ~/.lewagon/project_data
	@mkdir ~/.lewagon/project_data/image_data
	@echo Successfully created folders

copy_from_old_structure:
	@cp raw_data/bbox_train.csv ~/.lewagon/project_data/bbox_train.csv
	@cp raw_data/train.csv ~/.lewagon/project_data/train.csv
	@cp -r raw_data/image_data ~/.lewagon/project_data/
