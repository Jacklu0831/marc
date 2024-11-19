VOLUME_NAME="marc_checkpoints"
if ! $(modal volume list | grep -q $VOLUME_NAME); then
    modal volume create ${VOLUME_NAME}
fi

git clone https://github.com/ekinakyurek/marc.git
cd marc
git checkout modal

# run the modal
modal run modal_main.py




