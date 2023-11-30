#################### PACKAGE ACTIONS ###################
install:
	@pip install -e .

reinstall:
	@pip uninstall -y facetally || :
	@pip install -e .

update_raw_data_from_GCP:
	python -c 'from face_tally.interface.main import update_local_raw_data_from_GCP; update_local_raw_data_from_GCP()'
