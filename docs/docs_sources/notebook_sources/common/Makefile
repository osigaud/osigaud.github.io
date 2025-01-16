.PHONY: help all student clean folder notebooksync rsync check

STUDENT_DIR = student
TEACHER_DIR = teacher

PY_NOTEBOOKS = $(filter-out $(wildcard _*.py), $(wildcard *.py))

TESTED_NOTEBOOKS = $(PY_NOTEBOOKS:%.py=tested/%.tested)

STUDENT_COLAB_NOTEBOOKS := $(PY_NOTEBOOKS:%.py=$(STUDENT_DIR)/%.student.colab.ipynb)
STUDENT_OTHER_NOTEBOOKS := $(PY_NOTEBOOKS:%.py=$(STUDENT_DIR)/%.student.ipynb)
STUDENT_NOTEBOOKS := $(STUDENT_COLAB_NOTEBOOKS) $(STUDENT_OTHER_NOTEBOOKS)

TEACHER_NOTEBOOKS := $(PY_NOTEBOOKS:%.py=$(TEACHER_DIR)/%.teacher.ipynb)

SSH_HOST="master-dac@master-dac.isir.upmc.fr"
SSH_PATH="public_html/rld/advanced-dl/$(notdir $(CURDIR))"

JUPYTER_FILTER=../../common/jupyter-remove-sections

help:
	@echo "Cibles possibles"
	@echo "[clean]: Nettoyage"
	@echo "[teacher]: Génération des notebooks (version enseignant)"
	@echo "[student]: Prépare les notebooks des étudiants (dans student)"
	@echo "[rsync]: Upload des carnets (version étudiant) sur $(SSH_HOST) / $(SSH_PATH)"
	@echo "[check]: Exécute tous codes pour vérifier"

check: $(TESTED_NOTEBOOKS)
	@echo "All notebooks tested $(TESTED_NOTEBOOKS)"
    

tested/%.tested: %.py
	TESTING_MODE="ON" python $< && mkdir -p tested && touch $@

clean: 
	@-rm -fr $(STUDENT_DIR) $(TEACHER_DIR)

folder:
	@mkdir -p $(STUDENT_DIR) $(TEACHER_DIR)


$(STUDENT_DIR)/%.student.ipynb: %.py $(JUPYTER_FILTER)
	jupytext --to .ipynb --output - $< | python $(JUPYTER_FILTER) --exclude teacher,colab > $@  || rm $@

$(STUDENT_DIR)/%.student.colab.ipynb: %.py $(JUPYTER_FILTER)
	jupytext --to .ipynb --output - $< | python $(JUPYTER_FILTER) --exclude teacher,not-colab > $@  || rm $@

$(TEACHER_DIR)/%.teacher.ipynb: %.py $(JUPYTER_FILTER)
	jupytext --to .ipynb --output - $< | python $(JUPYTER_FILTER) --teacher > $@ || rm $@

student: folder  $(STUDENT_NOTEBOOKS)
	@echo "All good"
	@echo "$(STUDENT_NOTEBOOKS) updated"

teacher: folder  $(TEACHER_NOTEBOOKS)
	@echo "All good"
	@echo "$(TEACHER_NOTEBOOKS) updated"


rsync: student
	@echo "=== Synchronizing student notebooks on $(SSH_HOST):$(SSH_PATH) ==="
	@ssh $(SSH_HOST) mkdir -p $(SSH_PATH)
	@rsync -azv --delete-excluded --exclude .ipynb_checkpoints/ --exclude videos/ --exclude .ipynb_checkpoints $(STUDENT_DIR)/ $(SSH_HOST):$(SSH_PATH)
