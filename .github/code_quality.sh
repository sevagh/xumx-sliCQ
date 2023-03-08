#!/usr/bin/env bash

pip install vulture black

vulture ./xumx_slicq_v2

black ./xumx_slicq_v2

#pip install perflint pylint
