install:
	@cd spam_classifier && pip install --editable .

restart:
	@rm spam_t*
	@rm *.sav