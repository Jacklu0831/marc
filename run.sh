VOLUME_NAME="marc_checkpoints"
if ! $(modal volume list | grep -q $VOLUME_NAME); then
    modal volume create ${VOLUME_NAME}
fi

if [ ! -d "marc" ]; then
    git clone https://github.com/ekinakyurek/marc.git
fi
cd marc
git checkout modal
git pull origin modal

# run the modal
modal run modal_main.py




