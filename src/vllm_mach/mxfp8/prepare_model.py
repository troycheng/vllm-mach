# SPDX-License-Identifier: Apache-2.0
"""CPU byte materializer for the fixed MXFP8 champion model asset.

All checkpoint sources remain read only. Tensor/file hashes bind the fixed
inputs, not their provenance or redistribution permission. No calibration
records, private source paths or old experiment receipts are required.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

PROFILE = "qwen35-4b-mxfp8-champion-v1"
L0_PAYLOAD_SHA256 = "133aae145c4a21963520f538b2fc3a6cff263f3ccb128ba21ade6d91e63686f4"
L0_PAYLOAD_BYTES = 47186176
MANIFEST_NAME = "mach_materialization_manifest.json"
LAYERS = tuple(range(3, 32, 4))

# Compact, redistributable tensor identities only. They do not certify the
# upstream model's source, ownership or the availability of the L0 code asset.
QUANTIZED = {
    'model.language_model.layers.0.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'fbbd1e743f0ea4d1ad9f5bd1bf56e35cfd784d71d24d6948dc7629ca83373182', 'stored_values_sha256': 'f91b8162de4377c12322eb2e12ff1609c301befef69fed3f5c403215594eb0ae', 'stored_scales_sha256': 'f21d60e7b6379b6da7b404f3b3fa929c426cad7137fc8bedcffe91052f166a23'},
    'model.language_model.layers.0.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '25af60b66fca36f2cce5ae08be73d55ed0ee1d51d41319ce5c79fced8cf908d5', 'stored_values_sha256': 'e4ceacdb33a565bd0a7d7d161f5659b2216a4747718207204e46e5fff689f1b8', 'stored_scales_sha256': '9b89742cae9edef4aeef9599e715f7db55c5e8a923945b7555228415453f713c'},
    'model.language_model.layers.0.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '3ab346e604ca4e59e8f0ec9801c785ba48761de4d4df741890e0b740be8df0eb', 'stored_values_sha256': '488940884998fca2189a63efcbc51b0932a1965697d1a118fdf4a41047ed6a9e', 'stored_scales_sha256': 'aa4612920af7e52557d7726495c0359f4f3d8c70835a9c7c00cb33796bda5f62'},
    'model.language_model.layers.0.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '7da9d1bccc5700c05e89305f53bd0dceb3f990a8632d39f86676911725344fe7', 'stored_values_sha256': 'b9cb7a9afc80557df932fc7f9322b8e8852bd736c33ff8e8ea48f2f998ee869b', 'stored_scales_sha256': '853bc783b31f8831a77a95ec8539a9437660d1de2d9667229d854e840cfee91f'},
    'model.language_model.layers.0.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '0cd58c188dbcd2507b89005ab99951922af3068cda69c0894a7d2c1f4201f650', 'stored_values_sha256': '0fee43d02622495e340328d22750956b4cdb252922fce4dcd91814b6738ec407', 'stored_scales_sha256': 'a9511289ab56b3746435a9a6f263a57fc41ab3bdd0429363f5d04075ca9cea3b'},
    'model.language_model.layers.0.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '15dfff35691a98fe9906aa6e57531092c7d1370bbe5578263aa58e467b724562', 'stored_values_sha256': '5a534a29d716a5831a02ab13a98393736688ad49ed8191cdfbc9f04be4795df4', 'stored_scales_sha256': '477918dce67bc084ce709e174979439960cf6f9b7e94292a60ac915b3e4f9229'},
    'model.language_model.layers.1.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'bb2f2c232305dd67aa73ea11b40c0c4b5519538aae4de07b8155385c98a2dab8', 'stored_values_sha256': '131aa69971b9ced47d41ac67528d80ae217f9b78223ff6e34437edddf2ca8b4f', 'stored_scales_sha256': 'e0135ee5d02cf1a5a54cd418975a970af0ea05e7baed768ad629b133d550e436'},
    'model.language_model.layers.1.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '6e3e3b54751a36a1e2565ea1618eca9ec9f8261f95f1d894c515f8fd30c78ed8', 'stored_values_sha256': '160e3f017e2c399412b2d307589b10367679fd33bdb855ac808fbd5717cc4c89', 'stored_scales_sha256': '2107ad6c4ea09117112b6dd53227cab40424603c7bc56aa992428e1c2dc9ec16'},
    'model.language_model.layers.1.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '56c195855b9907d2e74418ad1937f0c974e9caaaec7387480b9aaa81755d8705', 'stored_values_sha256': '4a0ececcbb85ec30164a36188ad3a7ef2db3a9ceb2a6a0ccf67fda43541a36e3', 'stored_scales_sha256': '5b38e9fb82966511a701f5e1e9f36ce6b69aa5fe6a0ce9491b11295d57487c36'},
    'model.language_model.layers.1.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '8de486b50183d8d8067155229567a506e30ee16c99f78f303a4ed4b6bc6aa3b3', 'stored_values_sha256': 'c34c83c5a8abfdf099847029bc3484086cf3d128fde683144f2e66e10aa3f6a5', 'stored_scales_sha256': '89dd146c62500e26d267a8257a7c21882162fa3a95c1524ff67652a4e457b6d1'},
    'model.language_model.layers.1.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'cd5a8524a291957da2426549273e72b52d6f41e1c44ba24b6f98769a95685e67', 'stored_values_sha256': 'dad6bbbc1bb03a9f14a7bac59c89df3d3fde5b2619f8c410a47a51f307f00d44', 'stored_scales_sha256': '45568533bfe80f0263cd8e426317299570cca703a255976384f2a7c266187b06'},
    'model.language_model.layers.1.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '79e28d83376ac523bfd26b7d97f05951a0654849662d7a07e27c5a979282e007', 'stored_values_sha256': '38e6d5b218ea41010ffa49c459b3907df3e3969461cc5af838af749286237ac5', 'stored_scales_sha256': 'ae043f1caea23a8aba04931194bf7f54fa0f79aa6c362a41916dede347f71e2c'},
    'model.language_model.layers.10.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'fadcce3591b07403a366b33b9dc5b2be274a9d8837ef17fc62d2e85492dd34b6', 'stored_values_sha256': '22b639cdee9197b58f59fdbb6fe1f9c13e0025b2b885d39e91ac2c936222e9a0', 'stored_scales_sha256': '425ee91897e5ab7a04900d13cfabf7c73f86106f2a1ee0d4ade980ac21fa276d'},
    'model.language_model.layers.10.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '2f8d7a7a2af8a9750296dd0c80bdc5f6dc71e69d495a3fde343b58c80189a4fa', 'stored_values_sha256': '94ccc721a7ac582e586231bc02f43e1a8ee9b24551a5b70796afa45096297bd9', 'stored_scales_sha256': '66123ea06d5b0b1a2f2a6c748411faeeab02bc32efe5e613e4af3dd36c7ef981'},
    'model.language_model.layers.10.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '7e56b60be13ea356db6a03830cf5cecff4e65c2f7d216e4bcdd0ed3df53c86f2', 'stored_values_sha256': '27d62f7bf6350f098303773a6ff787125abe568f51dc1f00e74d7207a8693215', 'stored_scales_sha256': 'b02d084e54097c84c3af92b6188f306c79df1d8450309250eda33ab6308fc7b9'},
    'model.language_model.layers.10.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'f61831071ffadbfbb99955b71bcebede278f8817bb5463e69be09600060a0251', 'stored_values_sha256': 'b3a5ae8864cc8aca4c17d4f8b81119714525bfd2dc5db43939d9385ffd8a3e94', 'stored_scales_sha256': 'd5ab5ec9122f15a3a367bc6b8c6f383efbd77829808e76740c78067879cf03dc'},
    'model.language_model.layers.10.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'e47bc28a98e57948d750dafa90e50e7788755b441e7e7f5bd10843155a506a4b', 'stored_values_sha256': '8c4576f4bae619133c8c72f042fbfbdf8de8efdceddb7c4ed37115cc406137f8', 'stored_scales_sha256': '567bb6beed8808a313ad8da15c82f3a1a70548eb64b536e5dd082e66bbbb7917'},
    'model.language_model.layers.10.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '98ee7ea008b5cb95c79bf941a0f7ccc9a5f8553f0562bbc5a190ce9bfed71cab', 'stored_values_sha256': 'b4ee070376aa9ca4f3bf9d3432e59985fe9e881d2a4a8056fd546a6197cd8350', 'stored_scales_sha256': 'b3c396592ec1470db91902da8df917e95a295a414f5b8b1f241aec30963ffc03'},
    'model.language_model.layers.11.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'e27d01259d11debada98c992a2064651161f9d42c87bd89625cde21c7732e806', 'stored_values_sha256': 'd712c55cc451a1c91e0b16e2d20a6cb30f073cc6bbd915d149a2b0fd538e3e60', 'stored_scales_sha256': '102bd2e52fb9e8142a500f9c091e62892b4ea1b6c30966ab4b16bc6957ee40cb'},
    'model.language_model.layers.11.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '1eb9891583639ea319c3bc58148fe4359a00db66a36638dd5ac44d912e2cf3a6', 'stored_values_sha256': '19076794bcfc78a5783a850b903b88394ab4c99f0e08cdef1a8098246f0c7212', 'stored_scales_sha256': 'a9c7bd580443e0b96acab6bbdfe9cfd6084057ff63ce6b135af9f0456dcc2720'},
    'model.language_model.layers.11.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '7e74c94bcdfb38496c986171d55e3552623d9f77a344b04a4c9d93168ff939ae', 'stored_values_sha256': '2dd2cca4f32594ccd75a85db6d20b11e23974e2679f3e7c1d34704a1fd8abc00', 'stored_scales_sha256': '5a1a1b5797d407c1c4148ebc542b6655b71706171cbe58a7aaec31cf3a80d7c2'},
    'model.language_model.layers.11.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': '38cf8e3c7502adbe05a4cd60507274faef3d40dc6d79817a7e7f4280690991b1', 'stored_values_sha256': 'e64c36bcc4eaa617faeff75b71816f718dee267d8a2d348c4e72c3605c35cc38', 'stored_scales_sha256': '9d73443f55d947529bf8e01ba0a4b2e67100c3e5a9b333639fd38cf4e614ea3d'},
    'model.language_model.layers.11.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'a38f142132d43de5f908ca132321b4c7b036d7a9792792f4519c56abb2b0c19e', 'stored_values_sha256': 'bf7abf8862b6098eecbc432ff5589b15df2962f8acd17ea4ecdb98c01b31af6c', 'stored_scales_sha256': 'c70c3de8d29417f110ed134bfba0445b92c099da640d8b7ae469922f49a78a29'},
    'model.language_model.layers.11.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': '1326462da94a30b4b57dff724f83815cf68b388b9ad2ea431b459852eb360d18', 'stored_values_sha256': '0a6eab1a6f05a2f6a0c210ab4d96fd49a9ac4c5ce497bdd56b0617195e72386e', 'stored_scales_sha256': 'a4dcb1f38d02778eae52a2df721189824185ff224e8f3f3204ac9d59ced56edf'},
    'model.language_model.layers.11.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': 'c3614c7e1714a51aceb58e6871afb4358dee211e2dde82ec438c14cbd0cda8fc', 'stored_values_sha256': '0cb37aa842f1b0d3fd1e2ec07a9bb22ecbb4228225a0e0ae6ec652df09022fc8', 'stored_scales_sha256': '1a470f714054d5bb40a47f2332f7e71080e38fece2a2b6d0f4fcbd5751d9e027'},
    'model.language_model.layers.12.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '5b655c93adac07150ebc72f8df0514405a525ca908f8e774d841c64fa4676b49', 'stored_values_sha256': '0d91d16380604be4c88bb0b2f2da2a245f714de1d485f92836011dd86f167c21', 'stored_scales_sha256': '032e626d97fbf71be8af47c7578d39ac463ac4d86383ea4245ec90508f65f482'},
    'model.language_model.layers.12.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'f51e23fcae7b2be420c02b615bbcc99c4f3c23bee0a13ca5778c5f2563e7e282', 'stored_values_sha256': '075f434b70aa87409f24525ec6047576a6938110e617a42e23dad1f70641b886', 'stored_scales_sha256': 'cfb53a6d087f13f8d984f793b630b15b0e990965c95e95646cd4490eb3ea6ac6'},
    'model.language_model.layers.12.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '9113a2f78c772b1fb3ad12fad58027b1cf964523e16c9d9b67ae0ecf99f02faa', 'stored_values_sha256': '0092bb0da6176a45f2af0683f1cd751a1ed1134d624e5d6d34cdfbab0bf6d262', 'stored_scales_sha256': 'c83d403abbf631e47c60514d05a0580c748174502023f44b42db56149d2dde51'},
    'model.language_model.layers.12.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '4115088bfa678fbdb65bf4a05f9e26cd0f5dd85d2b708ed23ad187d442bdd922', 'stored_values_sha256': 'a1038fbf49a84a6b1d4587ca8756e7115490a2a332307e227f14d47fb6f3d0af', 'stored_scales_sha256': '971a4574d5c578043ed05608b2b55642538bae71cf8531bf1a810754678bb7ad'},
    'model.language_model.layers.12.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'a3df503dc8d35b999804f52ab89629b501784f97e58abd55952defc9bc391148', 'stored_values_sha256': 'a27dabc229dd94906919f432417c721205e96bfc3bc250bc51f88ad410cef173', 'stored_scales_sha256': '7a1039770f5b0286609ee724d4853cba7ed936113c10e6e02d35c8e02d199248'},
    'model.language_model.layers.12.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'fb21e56a416a743729069878193cde548be60a8f3e2a6b88e2fefbc457addf9f', 'stored_values_sha256': 'da419395951bdaa52b19301315b953ecd3248bbe3965303e9a892a06cae37d94', 'stored_scales_sha256': '7c44df9e13c693db84143867ddef446dc0cad9ddae5c14eb032abac1998c3d70'},
    'model.language_model.layers.13.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '9c2f8f486c68e5e9b1b96b2c3ccb6f88ea0225e22302acfa58d7cea786736b83', 'stored_values_sha256': '7a9a0aa1612b3023dd92db62b9f703cda6d7f2e96233bbe6cee30365109d3e33', 'stored_scales_sha256': '57eb713636ed79e48c4ab1fa3a013f74c5f98a88e004bfe8c3917d518a1995e5'},
    'model.language_model.layers.13.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '39afbde2b59112c9f7c788208d1aedb58c5a10ebd1e4f0d5df0c2497169ed55f', 'stored_values_sha256': 'f72abec00bc84f331343df0be1c1aa3fe1c2569fb5b0a188143df2f310f8b187', 'stored_scales_sha256': 'dd45bb494400f12f31c2b4d53d21fe7540449e2ce216e0eaf70b3991af21d39c'},
    'model.language_model.layers.13.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'd12ad740d220fe3d368b95e853fa36d1cf16a976f83be56ff5f1d075c4880cc2', 'stored_values_sha256': '8da90d865558823b61246c0fda3560b20412cb6943de3dee0b3532c68bb8a188', 'stored_scales_sha256': '9be68824c9665f3a197f50d1b31953b5896a04469ae3488b4f0c09377423e5ec'},
    'model.language_model.layers.13.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '57a1c0dbde6631d2715f59b5f37152f58c37ec16daab2077eb001f69fa614de9', 'stored_values_sha256': 'cf52646163e0abd848d2b1bd0cbda443bcffa2529bebc3541f2f9446c92d7e44', 'stored_scales_sha256': '6ab928f6ab2f8870af3d21185d8b9687242243a9cb6b3fe7e0f09d8cdae5afac'},
    'model.language_model.layers.13.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '7e694946ee5396d0c279f54a8da1d01b6651dfb9198a65aadab50d7582a5a4d4', 'stored_values_sha256': 'e68d9a59c44f4749aa71a2a271e0ea141a520e539854a43aaea435a11fa787e5', 'stored_scales_sha256': 'b8a634f7187439e395843ddef9d12535b1dd244d136f19e75de47a3895b0d770'},
    'model.language_model.layers.13.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'b7f6930841d959cf1c06082f7671bc31df1216e620e1cd2ca3f36f04ed58475f', 'stored_values_sha256': 'a9a31ad9a49bd5797d86632350c8b5d379ef184a118b06cba62e1dcd71b97d95', 'stored_scales_sha256': '889832bddd43adb34bfe9ecb846e0c2560cd426cba6870ae527de133d9225593'},
    'model.language_model.layers.14.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'b58026b3fcd00251765ae7234d9227284c0e25d8366638a9cf4dabe142bcf8bd', 'stored_values_sha256': '67b7492641d006dd092d92717b6d907d2d461416192f5fd53c774808ff837129', 'stored_scales_sha256': 'f3cbf8c292a14c888773e031b0ebd313aad9ba5d72985c05543e63c293ecd5e6'},
    'model.language_model.layers.14.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'db0b0cd9130bb4923872936617ea4dbde34d484199ff8db6e45d791e9831313a', 'stored_values_sha256': '043542d5959ce95f14caaf08f96c1c77ceb00b7444273a61499dd8088a9e1249', 'stored_scales_sha256': '2f435f8d697d8d2715bbacae9c2f8bfdd13d55daa625b118486ef1f0c5444c27'},
    'model.language_model.layers.14.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '1c0b3f88cf6745b1c78445e12bbcbc0352cdf970e3d6813b05a495d018dd5eb1', 'stored_values_sha256': '2094ff0a72fc9c6cb8f697d7f8efec12cc3c8b0737ccf9ce4ad2e2c6a4fbcab4', 'stored_scales_sha256': '2738a94500aed934e340b0d8e7117728c134fecc8418cff45a7f25dd7371a40f'},
    'model.language_model.layers.14.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '0169cf062636bb6ce00cb4d367cab66a24bf2adcb107d2c4f65435a693f9ecf2', 'stored_values_sha256': '8c3fe23bb5748f49265f6410cbf5dc4a41810ca5e2efbdd254bfe07f3b48babc', 'stored_scales_sha256': 'b41e4f8303c601f44012741e8551a2b72ebdd221953af1636c5c0e77258401d0'},
    'model.language_model.layers.14.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'db6adc5dae751b1f64134a1dc5bb88460ab2d5ba5c93e061cb1d69cb0a73bd82', 'stored_values_sha256': '2cc375ae91c5ddca83ba4c42a0fccb44063609737624f5e35020ad7f94eba317', 'stored_scales_sha256': '9635c74fac0096facb0974947e04bcbdb93eff13834b07e08b095474c91f0fd0'},
    'model.language_model.layers.14.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '0548b0c7dacf4da181b78719cbd3c05e5ba547e5d8c19dae3e41b51d2c7460cb', 'stored_values_sha256': 'c8b04c6f9222e52fc1829ec90fa1f6e3ef0dc3546a1be666f0e54e71546185f2', 'stored_scales_sha256': '32919f09e3dc23871fff12e6cc4f20ede84ebc67152f1ebea02b753f2fc839e3'},
    'model.language_model.layers.15.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'ebc53f36a34e6f3c3c6e5f54b955a1566f3aa1de43d3e1060fabf845e1e30fce', 'stored_values_sha256': '1f66db4c1cbc8c4b74f7a664ac69ccdc09587da6f3917b9df3d63c5443510443', 'stored_scales_sha256': 'a2cbd26b6e6c26f7180da094cf75285eee767dffb764d094e8f03e9d299af7a9'},
    'model.language_model.layers.15.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '79387d8c173c16a0311cf78fe6cf3ea43bb9ef490fbbdb01d91b300b8eb43c48', 'stored_values_sha256': 'a78720d1abe5c9f5c18cf20de852a90df6bf6c84d94b5b9f21477dac1eafafe2', 'stored_scales_sha256': '2fcf06e309bc50b4f61db3ae5b7bed7869ca22b1ca673e36438b948f117876fd'},
    'model.language_model.layers.15.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '42a3f5e3b517d58ad6bb0c859d562def8f002320848c542776d034d57af6d008', 'stored_values_sha256': 'b142d17114728e91f6c803af67fe1165818b85946c73a0d49b3ffba8460bc336', 'stored_scales_sha256': '6f4bc096d5c69b908d84b953130aab2a34f09025facd6c1c654d65de63718c28'},
    'model.language_model.layers.15.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': '1356092f734c5f9334b43881f2269f3d120f9a3268eac3bbfef86fb5baf57f65', 'stored_values_sha256': '67304db907ab32e89114a5192d027b07e44ce91f11e3fe290efda3a23f084940', 'stored_scales_sha256': '82ae593b71527b7e20aac0ecf94ae4d5283a926aacdecb4cdfa1455acb1a4a9f'},
    'model.language_model.layers.15.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': '0bc0060feb2faf04d57092a820714d38ac901195a13fff175199dfa9e1107ded', 'stored_values_sha256': 'a8640b2c33e5b19d0be9cbe9c9222800acfee29489d6787dc726530e7dc4a728', 'stored_scales_sha256': 'c907bfb2125d4af35f70a05a1a997a79e4bfe3ba058198ab5b2e905cd37591fa'},
    'model.language_model.layers.15.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': '6db9c395d1d15a26ea7c53fbd8cfd35dfe24e2dfb1efafa34ecc52a638f6fa45', 'stored_values_sha256': '649e42bd11d160a3a8ff2757d52ecb86f03088ed643daba6fe5435284033ba04', 'stored_scales_sha256': '9c20a90688dff7e7747836668762cb6f009702aa8bf189fc185562ef3925499a'},
    'model.language_model.layers.15.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': 'c01d2e81fc3a21235cc03c7d5ba64ffc6b30b95a887221a67975ce5ab51b1887', 'stored_values_sha256': 'e82b38f58a23cf869303157551d6d0a05e268f388c800f37337c89e558bceaa8', 'stored_scales_sha256': 'ea229a0ba02ba4c7e605548230ab30f0e726eaecf012ac05a5b6d2eb9a35b725'},
    'model.language_model.layers.16.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'f1c2fbac68de29b1609b28f65325b55280fc907484beed50addfbbc763ee96cc', 'stored_values_sha256': '93a780ab4ee0a1d07db8dc56bfdd216c1d0ca782af6269d16c7aa6bbaddcd558', 'stored_scales_sha256': '2021da97ba78466a7d4d8aca5ccf7c486eafbe50da61c7f40906f7d405510923'},
    'model.language_model.layers.16.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'b5f27cffb91efe8485ccb55cb4b668de1881a14eca84c0f7a0192c49e1998e75', 'stored_values_sha256': 'd375588b7ced70467e051b376beb0121be4f27e6879f7ac05f7baec47ef7162a', 'stored_scales_sha256': '162f1e40a6966927bf0a674a0f13671cc24a107b2b94c435edda56a7c32a57a4'},
    'model.language_model.layers.16.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'cc817e04a3a34d798b10ff1289c031da44188d4e7f7c75d5a41928f9e8b0bb60', 'stored_values_sha256': 'c3d3c20501b5ef25bc202300ab80c3cf9cd588e347d2c6f09c06629064d25786', 'stored_scales_sha256': 'e82e6e91bb7ab8b6a56cf155f009a2e4452c96bffd224b1fc0a58d54ecfcb200'},
    'model.language_model.layers.16.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'dedaf4d7835fe1319173ef431e0cddeac775bd61792bc17ebef5e6938f54d395', 'stored_values_sha256': '25a5dc0997e6570b3e2b6ffbd95e6ff9bc2687a15aeed2ff8a60abdf94d19bfc', 'stored_scales_sha256': 'aac7212af48ba96a5f70070e0a4691ff9187ab4b8e91323adba9e1ba4ea3e867'},
    'model.language_model.layers.16.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'eae9a2d75397eece72894059c715557d0a070d6a4013d5d559156248b2b609fc', 'stored_values_sha256': '3def8b530d4d6657a57538a7f86ccb1d14ee8b90ad72f9f6fce324a32c61823f', 'stored_scales_sha256': '09b92359bfa677ba6843f2367804bc79e630314f3e5230b223b4643c1e87cd01'},
    'model.language_model.layers.16.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'ad0f6b3f64ff161e5a1273b453b06e8d42823258838596618529e9f549cc2147', 'stored_values_sha256': 'ce0bb0338317fbfdd8ecef1f3ee2d4883ad03a3d9f8c32ef4ee72fd15ff33a64', 'stored_scales_sha256': 'f175ed117ccfb3b649914a0b07d3a212ec65957e499bb0ecbacebd0e02387cc0'},
    'model.language_model.layers.17.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'b9cfdcf84fe063eaac8db43b02f77d4a7f963c151430597a41e33cb934dc00d8', 'stored_values_sha256': 'd585c15bfe847c276f7291cf10271bd236c89eab69c9b6383f3da2834c98b966', 'stored_scales_sha256': '6857cad6aba99fc99825631dcb5e6180d189c689a7d895497df52f20b53e93ef'},
    'model.language_model.layers.17.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '199a9f95808cd20a602e3179c6969dc43ba965e970dddab08a217baa6651b6f3', 'stored_values_sha256': '91e56781b080f2f797f0c725c4e49b6e1570cde6a7384da72a21035939f31cb6', 'stored_scales_sha256': '06cdaf7ff73b79022f8ff1cdc30613dd7317ba064fcce55f7fb887876a703857'},
    'model.language_model.layers.17.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'bbf5be2b5ede455a8c5dd766094d881cc1fdc1c5d4d1c17efe5d54e26e47bde6', 'stored_values_sha256': '91a7c9fbe9b8c1cef99d56328bfb97d73552643f92db0c285c7918bb65b31b7c', 'stored_scales_sha256': '73ea54980d0b554c77377872f5c170c1ed3e7a47997437ba491cf95303c5c4bf'},
    'model.language_model.layers.17.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'b7e8b3f427211a28f855d5a8c8c7b675489df25693d3df41f11322ff30ef2e88', 'stored_values_sha256': 'c95ef99f760d4ca108fa75d2157b480d44d253bd95d85d2429f1398d9c49bb83', 'stored_scales_sha256': '1b45936e86692135a2dc4dda36571ca927fb8d4118eee3ef5d05c197b21c5556'},
    'model.language_model.layers.17.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '7d3b63a24ce4bf8fde9ba1c68240fdc14aea70b43bf17c6dc70036d477cba084', 'stored_values_sha256': '22eb0dbe50b35fdeb8cf56d58f6d14cfd74bc49e42ea4e60864cf3ff558e60b3', 'stored_scales_sha256': 'e32e06ba695088f1b7de3b0544f5b0724677a7e57a2053d7fd9ae84ecc801edb'},
    'model.language_model.layers.17.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'c7dad8bc980f0c7a1ed729e9da1acaa9167e41af10d5c2023b62a33a82ee0d10', 'stored_values_sha256': '9cf97bfecb3716bec40ba77efb1a0a1b54edb85981e2855724822be9065713c4', 'stored_scales_sha256': '97c5d3ddad5e2ea7e1a2af420479202e4d49ca3d8da1906975038aa3629b5466'},
    'model.language_model.layers.18.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '6bfd4ce50375bb3f5e0253b7bdd577d4ccb6f3ee6468afa48f8d9a1104ef487d', 'stored_values_sha256': '3f6849cbf6dffd7e4a222ef2ae45af8cb7020a2ce869b32d62bc1e3271184f5b', 'stored_scales_sha256': 'b58ceb6d0464cd74213ccd873e31648a1a99a1c30b4fd97f70a5a351fee572de'},
    'model.language_model.layers.18.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '4d46686eba22c9893802bc79a9073cf2fa9937407588c48f17a502345a9aff7e', 'stored_values_sha256': 'f70ef4aae6eb01a501254e2a5ad1c0e1e351dc324814f95010fa6aca91b2e709', 'stored_scales_sha256': '10b2c983222e76cc47e93f636b5cd4db9d0a025d6687504ea225d8d4f0e7172d'},
    'model.language_model.layers.18.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'ab6e88983d10f31c9757e4083812676e17672329c0c9cb1edc1133ea500bdc80', 'stored_values_sha256': '16dcb76df6bea0bdbe95026e92d4061d216ef152c75cfd6b19456e270ac80a3e', 'stored_scales_sha256': 'a2d9026c62cd20247575cf509a5c8cb72395e083b3a373ca7990f67cf5a4ef98'},
    'model.language_model.layers.18.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'b1ee1e6f8e347e5cbd04a7d0c71da5de4d32969a7fd6a24ad2663815df7b033f', 'stored_values_sha256': 'a5d75b0e30fc253f8cd60acb6fbb6b85c5b1eb99eadac19839a7711eaa1e1a03', 'stored_scales_sha256': 'b51c1e6347f9054360a0271758d9189186d36e1347a7c43ebcf6ebd6b7a346f4'},
    'model.language_model.layers.18.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '334baa850f7c3d95f00f98e9a9557df22843333dd85198663eb675975e2fd799', 'stored_values_sha256': '7c033adace58e193e18f60e26379c36df1a5c9d8ca1db1e17e3f20be3178d93e', 'stored_scales_sha256': '477919a5f6feed037ea8cad931cdf101dc22247cd792a8f192c8399a520d879a'},
    'model.language_model.layers.18.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '9f5147ff38603f98d93a0ecb2bea514d47ac8e90f829fb0063ac318ddc18f299', 'stored_values_sha256': 'ab60cbd81371a557180bf06558c87d659b5dd2b7d442c5f5c5740ed1a3d5cb7f', 'stored_scales_sha256': '6cf05c1f3559049f62162f03a3d45769f71e05c293b0afcd9db0e5536930ef5e'},
    'model.language_model.layers.19.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '6a03103aebef7c1e8474d031a7394e7fad7839c9aa567433e8ac7bc55fe0bc2b', 'stored_values_sha256': '5e6318f9ebe2b3a7e4984d953c514af0b0a15ebe86d88180c1a08d05aa1f3e57', 'stored_scales_sha256': '8047c2b7210668a7f9a61b3aa8248fdf57bd518c671575a1a60f172d6cb125af'},
    'model.language_model.layers.19.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '8643416c1220958b4e3098d5b43da5a06bc10a1e003b352946663f3113a6f335', 'stored_values_sha256': '68bae11bd4950d6e97f9bb57214b1813d978447ea34749773984ad9efd22c137', 'stored_scales_sha256': 'f46fdeefcb55737dec28db5b0607a0b49afe274d060cfdcffd5e7a21d724a3d4'},
    'model.language_model.layers.19.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '057e846ed1c67544743f6630d7a5f11e2cf9388cb61b5772c4954ba19c878791', 'stored_values_sha256': 'b47534b784dcfd029a70b12c7597221e0de7f9e4d3fbeb27c1de876c01fd638d', 'stored_scales_sha256': 'b24b56f76b255ab8003d184395340ebaec4720f7267e81d5ca6fa2607cfcc2e6'},
    'model.language_model.layers.19.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': 'ed920bd9cbe9354e5b48c5e930ca047a80e8a192cfbcfa28820b173bad14b412', 'stored_values_sha256': 'df3f2a696a02e8b10f64479bba049c6f33862ce639bfdaa56733c464f9d51bf4', 'stored_scales_sha256': '0a5f338f785817530fba95e9b5d11997709c8d855664e03b22b5457d72ea2520'},
    'model.language_model.layers.19.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': '8f32aefb92a428a1bdd9e4146bca0f8077b5abae22223628fa7b6d8089e4bdcf', 'stored_values_sha256': 'c7e2e420663e1f94c6f9365c48bc9132d969428115f1e7a5be9e4edaefab59af', 'stored_scales_sha256': 'a55c068577c4e6a6e384e221e2069b5c52319b94cbc40873e3e93274eba244da'},
    'model.language_model.layers.19.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': '4e76f73c68425b9174ba7b40be981f6de8994e4c538a3e930a3df145a5e07964', 'stored_values_sha256': '1f9518efb90286bf71e81fab1627460ed7c25611d53a252f1e006b4c571bca01', 'stored_scales_sha256': 'c112cda8daed7390d2ce579c23d76b8fb13e1a431ab4e24b95ececb407cc5c25'},
    'model.language_model.layers.19.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': '911fcdded32d6bd3018f44714cb3e546f80e462ba2695d93d28bbb60fc37516c', 'stored_values_sha256': '1ba6fa065e813370edefbad97a83ae245f57251b8fda67c7b92eeb27694f20c0', 'stored_scales_sha256': '1e6ca1542093216571858bb91c039d62d1c713c07b4518fa6bbeb058f48feb9a'},
    'model.language_model.layers.2.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'ab27ca41a5657c22e04e01a1db0db24a3a426ba86910bfb77ef726342bf2abab', 'stored_values_sha256': 'a080c200a12c80189131bc65a11244577369751cd7d4a4662866b5a8c112e384', 'stored_scales_sha256': '14f284bb7e18a4bc9be8e2de391e1bb27cabf1c17cc2d65f2694fa9c393c12fd'},
    'model.language_model.layers.2.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '8577762b5ffa11399e47cf9f032e494c989c01222a63defa8bbdaff27c86e0e4', 'stored_values_sha256': 'df08305d99a42476f76a6e7b1f7047547db250608c9784fd55fc7b1de35f0268', 'stored_scales_sha256': 'b5d6993c575ed6ede4bd38b49c666d035f66c2edf912cce8c13282b81ef8457b'},
    'model.language_model.layers.2.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '9daab2193d198812105961242019bdabd6b57d5cd33a40193d770b37ccf4ddea', 'stored_values_sha256': 'c6e891b27030bd8fc323315315c6849d22d1e16edd46641794f309bb197bfeca', 'stored_scales_sha256': '36654a9797788dee3ff54fd72a2ffdc602de55a05c773e97a259787238740387'},
    'model.language_model.layers.2.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '804692640322f2e5e820622caf308311ba99405802c5f74920d94a9d62116238', 'stored_values_sha256': 'eac1c4ebb4a06d437859a370af670a5ae069206aab7f2cada88402e38bc9c2c4', 'stored_scales_sha256': '43e3e65598a1410a96ccff881a5ef9402e91f86086b5eea4f7aab4284a666e6f'},
    'model.language_model.layers.2.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'b5a9e7b353378051d0d19f1ed80165ae40b014d46bdf6af9d8cadddc535f13ee', 'stored_values_sha256': 'd2af8dd1edb149240a029d46c77b04dafca127dd908d9d2ba7c55a170a909a35', 'stored_scales_sha256': 'be53762f78dcba421c092d67e57cb536ed3eeb15d75391a76e18e1e1c14deab8'},
    'model.language_model.layers.2.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '55d57b886d0e34060b6eeb48da5da1a7e238f2eb9cdf02ac61b07808cc9fc4bb', 'stored_values_sha256': '28b69e239d21469eaec2990a4cae6c009fbd0999d14206936cfe732e27bb0bef', 'stored_scales_sha256': 'e1dcfb0c392406cca572d77086ac50e2f5c1daa21318910887a68b00960427e7'},
    'model.language_model.layers.20.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '1419919af3b99b4ea0b4c1c9bb43203dcfd13e3090cbcd48747bbb632269ff53', 'stored_values_sha256': 'd269ebc331d2c4cc8b745ca496a93e8e3ac4ce0fb676e5a39a65505c053be4f4', 'stored_scales_sha256': 'f12b2047fb3c4225d08ccd3a0728baa540779da2e3ff88afa079c3d0d0192089'},
    'model.language_model.layers.20.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '0ced68e75f2960cbb252afb9e8aba387f2ef3803fa682525d85bfab7e28437d4', 'stored_values_sha256': 'b2b923c193c822e3af012f714f3c8372d21a1191451ccfbd471cc33e6d0078ea', 'stored_scales_sha256': '4c56f055be0b6f499ec634400cad0b974faa2c7c30b6f4e87b495f906592ac89'},
    'model.language_model.layers.20.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '898bc582ee521ef3c96e1734cd8832f2339e404845ba14b3eef4fe0723cc1cac', 'stored_values_sha256': '2c623d6af4f5ac467856f4a691340185a8de678e752623a9a62bdf092974b043', 'stored_scales_sha256': '32e96bf3f9646e87afe744b51706ad5e28e1ee870898f5411a5765aa94b68489'},
    'model.language_model.layers.20.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'fc2c6b5f23925e0c0994a67cca9760587b9c63c2450177caf7d8ae0cfa548fdc', 'stored_values_sha256': '4f30aa900f3b661d5bf11169d2d7efa168e7170d13d5baacef52ea30a4a4f9c8', 'stored_scales_sha256': 'b6d6b9d4765dadb9fa680b7511a948e61873b265c7532bc8c8e028215042bc8b'},
    'model.language_model.layers.20.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'bcdc7c89170ebadf837ea6cfe82bf1b1e456a37e4ac8db408ca91dae01b04331', 'stored_values_sha256': '897d97fa62c2c77c63fe64e33679975fa09b81ff61a06d043adc3c9d69b4f381', 'stored_scales_sha256': '43d1d704bedfc8821a5dd26e5256bfce29f3ffb356a1ac18ebd863825afde3e6'},
    'model.language_model.layers.20.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '9dfaa23b9ed114ba6891b6610992e53b61bb31bd91f7bf0151c09754ee148cce', 'stored_values_sha256': '04fe6e14787bf3878cf7c327ae00140193c077a4d9a1e0bb38beff4a643a35e0', 'stored_scales_sha256': '5152fa23c46c2107322a2f1aaa0663dd610e6f4ecd673e569cda1c3c351877eb'},
    'model.language_model.layers.21.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'd186c228b7f1ed0174d9994d99b1447204bc677bf26524ae610be99e9fe279ba', 'stored_values_sha256': '795cd52ffce70e88b8972d59fe084d14c705f76d60d90229110e12ea9e658b3c', 'stored_scales_sha256': '1c27931a782eacdb8ab20031603b8f466d67a9c065db09ff421f75b04e5fb419'},
    'model.language_model.layers.21.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '16744530503a98e3f6112dd87afcb911207b2383252be0b2fa32ffffd8294e44', 'stored_values_sha256': 'c3af2b312f1f2c6ec32d690e0a4dcc70dd01ac58b0af76d24f3008608c097125', 'stored_scales_sha256': '7d4b06a5bc0a724d32478f3cb8326013a16136375549f9739a505233dddbd490'},
    'model.language_model.layers.21.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '7d6a5261522775d038be4a5a2f92cd722c4ff2be24d4f0454fe4fab1f9242f67', 'stored_values_sha256': '1e25b0f9bc417979c0091a491f828931a634cf918578e03da656a426ff4e7acf', 'stored_scales_sha256': 'b327ff9cf748c03dd7b5e8fb22526fe4fe3e0c5ee9cc93e6cfd52fde4a2ffabe'},
    'model.language_model.layers.21.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '47c92f663744b78a1c1dcb63b95b6bcc4e5eb98713796c16112bc0c15d0200c3', 'stored_values_sha256': '352d829db7d7bfe638f349b16ccbd6e1f2b135d7acadbd8a077ac8ada6399c37', 'stored_scales_sha256': '38812c7e9decdd1643087425f0231cc142c4a03cc6046dca5b3a6f51465272b6'},
    'model.language_model.layers.21.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '578f53aa78863945ac5061292431b1d5eb9c265f891fec3d3741242581ee6f55', 'stored_values_sha256': '2964b6004c3eac4c78a5083837454add1a9689ef3ff6f9448f76fe7703d08629', 'stored_scales_sha256': '56fdeb72d688ed789ff6c5ee49c4f13a57db4dea31bfd31f34648f65d1958d3b'},
    'model.language_model.layers.21.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '463b731285b6fd03066613100093ac48d69614b03a257db5cd8e09ec89f82228', 'stored_values_sha256': '4b4bf8ec37b89ae443ff2cf9b0f31c4bd4ea28e008a10cef83a7ae47549c3d71', 'stored_scales_sha256': '21e8fabf7c9ea5674c9f3339963c3410a147cdb51232ae970d43a6bfa93e9c43'},
    'model.language_model.layers.22.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'a21c0a0c968e3c7853a7fb4326f86b009dd3d65dadb9edc4e2581e70a1171097', 'stored_values_sha256': 'fb3101f303b0b17e5e426e3181cd8795e4713c24c09e5f5887abbf848284531c', 'stored_scales_sha256': 'b827c5b9e0fbf54e006378196d616298fa0bc6453f74fa544854d6a4f1081a24'},
    'model.language_model.layers.22.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '99e3f15b7042a69a0320d4440a845fee7ad979cd8d7f0267b4594a402c7dc165', 'stored_values_sha256': '77850b2832bb16b7fde84c6b29f8cc0cf0bc5ca8306fbfaa050e09f9f2e07aad', 'stored_scales_sha256': '62bf2c63ddc31e06341df7321792be198de3c7a68aaba40321241aa6e1cebba6'},
    'model.language_model.layers.22.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '3ef307565dad9196eab3708abbdd64f8765652a2a885aa1f0873de795a788ffa', 'stored_values_sha256': '95b0a837e1f0704fe8ecb0d7ea18f7c9c6ff9af429eabd0d16c3e0289bcb7b47', 'stored_scales_sha256': '675216c30394e35fcafcb9dafdc29304e8c399b2c7dd9ca2503add8f2426b207'},
    'model.language_model.layers.22.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '40e77bf8d723d5edaf6ed1052633e434ce723be676e47baf799790c50af4d19b', 'stored_values_sha256': '59e851dd62dc7a60cb3a6db38784a1454c6c36eec045573d7f50b7703dbce35c', 'stored_scales_sha256': '129cfb91d8a9a2100cbd2dbe3a809eb65108c223e1825b2a6d421faf6cbcd982'},
    'model.language_model.layers.22.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'f84536e5c16c70e5ff1680824d1d6e0ff9871ba7c716f5e20dc931fd7c4fa2d1', 'stored_values_sha256': '70302d002a28ad42af8e07463b44d159866a9cdef3fa39f883b320c9fdf328d2', 'stored_scales_sha256': 'd6fcf73d6e20187a209b4f0bbd14cb5fb442fa2f34e004c0d3fc6f7ad20d75e9'},
    'model.language_model.layers.22.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '1b7dd6b24db48d4135162e8488d83a80248df88039903b8d96c805add755af01', 'stored_values_sha256': 'a02af720935bc62373562bad7503a5f70551f265aa66fa7378ac1f819616c0ec', 'stored_scales_sha256': 'f7b957e16733bc11a35c99ec6ef2717469de7ef81c24af6683ab1dcb64bf27c1'},
    'model.language_model.layers.23.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '5d097bec5dda4e01ed4b18fa4c400ed3a5b38546f2895819b9cd06bf4e9c2613', 'stored_values_sha256': '1a1a86ff4c9d9fe0a918422508ad9803be5ab1fb5d4dcfb3904e23ce6902d886', 'stored_scales_sha256': '968939356ff55ca0427386f6f095d84917a3546d3be12ce3bb59c8bc833ff850'},
    'model.language_model.layers.23.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'b233e95b899fb2a45c60e1e52ab35656922cbbe379c926f0531939e316a1b6da', 'stored_values_sha256': 'ebd6b30929f529131b35aa876f4f866e0f89c5c6ff5874e8c2b6f66f9cbdc6ab', 'stored_scales_sha256': '2e9de471397fdb55ba147456220247efe582d8cf31c701ae3f070651fd04b49a'},
    'model.language_model.layers.23.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '2b17760977bb95a2baf493b9229452bd0164878b7c47ab7b24454438b39ea47a', 'stored_values_sha256': '42c99e771c972885f9b935712a7c155d3f6c1281fac3cda1791fbd1a99a485eb', 'stored_scales_sha256': 'c94e070cd42589cd28b522eab7783c05dba33a30b0781f2f050a699158cf2bc4'},
    'model.language_model.layers.23.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': '7490a74a3a2d3eacb10f00781775a7f99f586ccb5dffa60497148f722ea22975', 'stored_values_sha256': 'cd0df3f8207324bc58ed15063c554ec2c1d66895494ef3e1ad311314c6d9f43f', 'stored_scales_sha256': 'f87f392f806940abc362c55d892c5e403abe4b47badee57d75dc19238128324e'},
    'model.language_model.layers.23.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'a147a5300c5211ee7457088512db0af98bf6e967f5bd30a29e2b245542ce94d3', 'stored_values_sha256': '67ab8fc9cbd1bd136e6349155a032b3e65ec64e468ddb74a6eab1ba0595d398a', 'stored_scales_sha256': 'ff2d4e1b693c3132d209d58da04663bff80aae4d0cb7e00d4c7a86eb4959d347'},
    'model.language_model.layers.23.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': '04fc08cd18df7416ca2c2f0c80a5446487e24b131651d70d675c7dc02d969b1a', 'stored_values_sha256': 'fd0acab9237d6e0af48c5b18d9964d8b174c85934724cbb97c1fe5f40a6c40be', 'stored_scales_sha256': 'ccce7cd4dc54e7934b01a0835e454a976786f87bfc506e750e0d4ac76bb6aa9a'},
    'model.language_model.layers.23.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': 'dea5387cd3cd6cfc9ed114f5b8637279d057d7a42965b6f9cc95d4f326480a3e', 'stored_values_sha256': 'bb47b2d11319b6245907f0853575b62d8d46a8193b0bf84e370606e667dfce9b', 'stored_scales_sha256': '9cdc79a66bd34974ce33cdfaf051cbe6253872330788f9ce7b3a145e93f26bf8'},
    'model.language_model.layers.24.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '478b749da200a76ba3389880b0040196b0b1495c98d9321d7c519ac082f3c0de', 'stored_values_sha256': '506ca963a580dc0bc7e23ae816023402a06b7c992db8bc4a8f3f7bff9006ece7', 'stored_scales_sha256': '1368fc7bcce3b825a6419b5183930a2dbafbbeacd2c2e50f33146b9caf77d235'},
    'model.language_model.layers.24.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'bc3105b60f995eacedb4410c4f77eac91cf474c2637db57fbd057528c04b575c', 'stored_values_sha256': '2eb71f1b15681b324dc98bb89c97e57959ff1c476b1e90e1b62a8277433196f2', 'stored_scales_sha256': '1993cf0c9b7496034896cf346bade17edb10393fc3b30754b96096e8ea7c384d'},
    'model.language_model.layers.24.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'ca302e1b6e9adb4817be002679972b4e22043d9c70943a30a98ce2f6aa78d3f9', 'stored_values_sha256': '1b2f792213b3a16ad597af345cac3e9972c4e2ded3bdfc3c1b1fb448b1635e14', 'stored_scales_sha256': 'ff03ad479022ac408011874eec4a0ad2137ea64359891a4a7cbc74ebdad8138f'},
    'model.language_model.layers.24.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '221751b9e47a2b475254ec23926bd1f00ef818c0e6e200688adefe00876cfed8', 'stored_values_sha256': 'daf1c3ee98056fc55656d38eb6a6f01d427f94ae81c658baebb10d48f3523677', 'stored_scales_sha256': '2f1f08d49b944f5e8680018866258c782f5a56f1ad668bc01290f426f999df52'},
    'model.language_model.layers.24.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '4ce56b334b80766e578c821ac624e367ad0dab5d9c091359399a27edb90bfcdb', 'stored_values_sha256': 'c415d9eaca3367cc7c71c4278eb38ee059aeec48b1cc77e2bccc9089eb774239', 'stored_scales_sha256': 'd09fb927e38bb05a202f2228e56b0c01368ab91080e511a14739feffd126d336'},
    'model.language_model.layers.24.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '54cd7866cbf120192b40e8dfce543d18aa85ed73b9e4552ba9a402cc9529847e', 'stored_values_sha256': 'f84ebd7f58a1de056024354bc1e360011f621bd723b461257b1bab9b9f0450a9', 'stored_scales_sha256': '14981cdac81985b871a19609fcf368a8b079bfa5c09cc8c7d1c8a32297728466'},
    'model.language_model.layers.25.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '45946866e7935b019365c4fb955da9fac87f20d0231570d8a988807c02836897', 'stored_values_sha256': 'd702ec509fc1714834bb4d5b8520c60d9b424f06245b7259f88006cf458a82b9', 'stored_scales_sha256': 'cbe0dfcea201ef4f38133b7171da01ba8c90159bcf9a54b1977944f6638cadda'},
    'model.language_model.layers.25.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'fc44b0705ef8e3fa50281b818999d74147b88788ada7c56c3d7934cc3b29b01c', 'stored_values_sha256': '62662c76227400e20bcbb82a9443d7c658b69a41a8097e347915cf2331b30045', 'stored_scales_sha256': 'ea287e9ff38654684af09612be28387aec83c61477817c20a1fd49be8ddfe132'},
    'model.language_model.layers.25.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '373faf5df801b2f5923a0998cfc7130118e135daeed21e2610126a39fe2a2066', 'stored_values_sha256': 'f760e0d7be2fdd69cc3a72f88f00266d47402ff7ef9f500a297c63aa9ac62041', 'stored_scales_sha256': '31d43c4fdf34b3d232d275fc20bba4df482495d821d062c77c4efdc2190bd230'},
    'model.language_model.layers.25.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '60d81543e69b78310ac4803c301321a0acf6a9c45aae325dfa1826b65d211f79', 'stored_values_sha256': 'e93ab413ec58582e2d0f9eb714d06a8fb80337cade552c426a74d075fd7afc05', 'stored_scales_sha256': '4a51338003047977b5f35a510491811b69884000d23c24774017c49920044396'},
    'model.language_model.layers.25.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '70d7899580974b58c33b502e3a6613686b4934b85b9e081dbbec5b9a334d040a', 'stored_values_sha256': 'e5deca748d5d67ca178194ba40a52e391fa1e59dd0b0dba74e4661e5573e1f57', 'stored_scales_sha256': 'c5449294c997a638af0e6d6014d8250d6f623d2bff4659fa2313d4deff2a4cae'},
    'model.language_model.layers.25.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'd50f750e38bc7ae37aa078e33ab39f28a8787d9c19e58e1eaac810bd964fc684', 'stored_values_sha256': 'ecfc0df91c09b873598cdb08ec261259de0ada56c603fef9118df0896bfbb3ad', 'stored_scales_sha256': '7a175aa1d81e0ef7a6554e50564d4515eb1ff8d5ac2d936a3039bdca52fdfaf8'},
    'model.language_model.layers.26.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '27012e9af636a59cd7b21eb2b540944a0cb3a28029d6772d17a4e43225d92a88', 'stored_values_sha256': '4227260d646d0c50b109023e39cbe0bcd2e5979bc4dcd73017b0d2f0b263fda7', 'stored_scales_sha256': '8f6f83cf657a52fe50bded08f9b2f5bae2c1cb77dd2f91fde4559dc58af71182'},
    'model.language_model.layers.26.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'b71f416b9dfdb1afb5a209cf7233f949eff6d385931eb37d8bc857fe954b60ac', 'stored_values_sha256': '3b10f5613ef2b7be99b5f83822231bcf1065857409fac82109b15ee61c4bd416', 'stored_scales_sha256': '8d444a2a1c6145000dba883786d18f71656048e961f45ae327406d1b7a85312e'},
    'model.language_model.layers.26.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '7371abece32ed8bd7460087bf5bd7c44b6da50052cb4617a45865a89a5f2672e', 'stored_values_sha256': '043b59caccabd27a3560b99cd82f5d83044b230043d650bc9f49d7ad7fd829ed', 'stored_scales_sha256': 'd569cbad42d1e000187da78db82fdb195ba94513dc179b64bab0309a241c3068'},
    'model.language_model.layers.26.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '897a54538a94172433a4f788aa0e979a268a2d1eba725754cd8e7cca5432bb05', 'stored_values_sha256': '324ec8638b2b1cc451dae81fa85a9a34ec5b2adc2e24d3918140d1477375cc65', 'stored_scales_sha256': 'a5d92f49a1826b1c6b889386e74e8b1c56ccbeac75bc87d3132b06698f9630a1'},
    'model.language_model.layers.26.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'f7bcc5d61eda03376494a4e9c8dd65fecfe1e1252351eb059e2530750f350ee1', 'stored_values_sha256': 'bb6192222b2f6f146e2da9903ba9ed2a1cf8353c85db95d03971220113801687', 'stored_scales_sha256': 'e3bd6e02f15b1f6085805f822616989b7a958d652378684455b0017b1f0b79c5'},
    'model.language_model.layers.26.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '15a62de4d44dd39b638ba97819bb8b11ca7db0b27813f810a9d887fdfec2d2f2', 'stored_values_sha256': 'cb43561b8fe4a3356942887f9937b72da83b9d229578e6076797ff7084aa2cbe', 'stored_scales_sha256': '47f9c81cee6600ff5fd895acf2036f28e32798b26ab7bd3dd3c094ca47f3d35a'},
    'model.language_model.layers.27.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'bcedbdd00d40a6dd7d8e150933092a7996c4e4a2a448ffcb1fafedecef5e3f94', 'stored_values_sha256': '00aac4a99b977970500bb6b9f8ca6d0d055b16cccc75132fddd435f401a81bb1', 'stored_scales_sha256': 'a86f5d4c3e71efd55862e27d574edfdaf56007d758add1741dc7a55a743229a3'},
    'model.language_model.layers.27.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'ad5d417b6742c8e5e828879006c0618a8e2583269f9c804969aff0a2fde11821', 'stored_values_sha256': 'ff27a7ce84fcc213c16cbbebcff0e327b2bbb3090d20a7b6e7f84c75a0cd90b1', 'stored_scales_sha256': '624c5efba8470db397a60492b45717f9c94ac02c9cb50140bc5a4a7310d1187e'},
    'model.language_model.layers.27.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '1de672732145547cb046e10ec1d38d1372f738889b61a4663d65cb84c02b6f8d', 'stored_values_sha256': '3d0fd578bfe0199ceb393f3fc57ce5e1b67791e5152a0386fd609b3ce0eb6c83', 'stored_scales_sha256': 'a65553eb4aeb21ae597f139e684127391d33d049f60fa599ddf91cec3413f47e'},
    'model.language_model.layers.27.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': '1f0e76e817c4354a176c66b3b17aa1865a987456ad32425e2b071fc2550ef38b', 'stored_values_sha256': 'e88b6dfa678ffbc3c5d95b6c2bbf44d48fe3a1d5dc4763c52ee481d742235e83', 'stored_scales_sha256': 'd883af21e7622f0a152decd101ea77b774e26fd187b47be9c319f373ab3cc307'},
    'model.language_model.layers.27.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'fb4896adabe8e3225de85ed6b9f86aa784b68b05ab1b6c3a4c15548c2b73c5bd', 'stored_values_sha256': 'dfadbc519216a490146d75eeae25752e6c889f31d6eaac2a00b1f31cc99e10e6', 'stored_scales_sha256': 'f9a10605f2c4fa94bf3926d621f97c7656ee6e3bf7708b662eb9e107433b814a'},
    'model.language_model.layers.27.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': '79863101469c652c44144a996ca5a5455edc4091a37cb5b36a73a777e78a52c9', 'stored_values_sha256': '5822e0a21e8bc39b2d5a2bb303da197a750bb5031905a4f98fd31eca91190bfd', 'stored_scales_sha256': 'bd813d4ed972b974af44e2f348daac3572a12881c323f123ea61407d0bf3c717'},
    'model.language_model.layers.27.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': '62da45d4536c732a481d82e6108d989b3147fca8335d5ab19cf6f43d9d20a7f1', 'stored_values_sha256': '1ec0a487a25d4023fae2d1fea892813608ebbb7fa13a863356a533a06ee25ca0', 'stored_scales_sha256': '400d2bb6f96339e01feefbe6f1cc8f1a0805b7aa6565116c7ee66c0569a6a494'},
    'model.language_model.layers.28.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'a5e91c73316944fad4dc8902c25b20900a3e86e7e2e562e6101d5fee45c24a62', 'stored_values_sha256': 'a08404b891ce109d8be8cb75fbf1f02524b60687bb3663c5618869dade8077b0', 'stored_scales_sha256': '13e57ac71eebab37d31d47e9ce79877401ba9b8fb879a17ec27e499161af969b'},
    'model.language_model.layers.28.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '058f47347464b17dbe3e3f316423c1d2ddb976fb5201a6ca38f951278b3bd1b0', 'stored_values_sha256': 'e435a5777d3ef284fa781f2f48c6fdaeaeecdbcce2299aeec8d9e0891a33c273', 'stored_scales_sha256': '723cd08bcb54238725aabebe9cf965c9cc59d53f527fafb3ba2fe5a57aaae28f'},
    'model.language_model.layers.28.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '2f5cfb2e6ba19d9f3c64c2d47669e85b12705a623b7578a277e7bac8281904ca', 'stored_values_sha256': '0016e417f084db7451e41a618b67831f35892a8eec6f80520fc722224cccf4a4', 'stored_scales_sha256': 'f2f6896d4d2ebc942e0f9ea1cd5c5a7b2ca64c2dbbb95a8d9268e167bd97115a'},
    'model.language_model.layers.28.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'f218b33e9f811a5a8ede7c3ebfcff16277efcf03706326e839f509d83fba3c63', 'stored_values_sha256': '1aeeb41b20c5021b20ffbce36ea5823ddfd857e2e6172b19e0d443b79606e3f1', 'stored_scales_sha256': '2e3b5e7b1ed2d727d6e6c678eb46dcfb98d4e6b660d87382ff98ef93677460d8'},
    'model.language_model.layers.28.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'd2f6afd77c9367504acf52324ab0f94c6387d88e81d096bbcf66086ecf219540', 'stored_values_sha256': '984a3f74b28d03eff6d5b6b4afb4b7c1c80545424c0c953c97b9ae35b0088e01', 'stored_scales_sha256': '912a7d042e5e45cc2d1e82619faa69cb8845124b7e1783d4dc60ee54ed20e55b'},
    'model.language_model.layers.28.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '506759450eb594cc2ecf478f9e10b79f14671f037ea5370bb70c90829b228195', 'stored_values_sha256': '8b0ca54080c4eb1a33069ac8d2780cd434a56a7d224876e9282acb05af685517', 'stored_scales_sha256': 'b32df386d5b3da5b1e9d7fad02f29e55889185cb723e3df09849cd195dbf2ebd'},
    'model.language_model.layers.29.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '421e86bd931865556b1e84cf91656729fbbc60e015153eda19c3a47b938645e6', 'stored_values_sha256': '8d24a349881b9a4f20acb74a394820ebbeac8609ee564a97c074f6810a1a3301', 'stored_scales_sha256': '07a7726d9f8dc8c1d89240dcf3505f18973e7e858b51155b30986833588960f5'},
    'model.language_model.layers.29.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '91b1d071110c1f71fa41c3d06db38638378088ade3baf3ff8d656350b2764fb2', 'stored_values_sha256': '165520c9912fc3348316cefac40c3486dcb764ae63302e79557aa0f3362bbc50', 'stored_scales_sha256': '9c79880d1fc71cd8ced7e31af91b1c98751a3249ad2d189024fb527a58ff7502'},
    'model.language_model.layers.29.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'f3785c7f8f1f60514964a2dee9e8f2fc8c2845c9c00b6e19d278301bdf125b49', 'stored_values_sha256': '03459d32c8865b7d3857cf766e037745bc71bf7d87f04c9baebd31292dd350f6', 'stored_scales_sha256': 'a34d70177bd854e8e059ebe406344f56992f919a86ef080c9104127d15cb89f7'},
    'model.language_model.layers.29.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '4953462042f04a84158856bda36716a2058abf8e5e5453e3bc218ef341de3c44', 'stored_values_sha256': '197252eabfd19d4d3907d22af772a16827ede12aec842adfdbab74a99de4c7f6', 'stored_scales_sha256': '119f2ca0d2deb3d65aa6e4f4893c34de38f93cdf186680d186d9b8f52bb8dcef'},
    'model.language_model.layers.29.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '5cc54cfe87c9acf5d3fbb971ce395ec0f47927b8f6d64cc5bf213a331df10403', 'stored_values_sha256': '0ccff6ccf00005f273b8af951471ac8cea807cdb29d5e35ba066a513a9da7889', 'stored_scales_sha256': 'dee592b96775dc4ad6e236287a3133c69f6d080cf5cefd0c93bfaa53232b8d2e'},
    'model.language_model.layers.29.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '0fdc1bb6cf93b89e0eb0065bd7e5deb67b3c4d8412b19b554694c0001cc80cbc', 'stored_values_sha256': '19934a31470284eeb94fcfa14e3cde5c5884f4e04eb6f9c8209129ec71863050', 'stored_scales_sha256': 'f18b80bc212e98dbc15f8f7bf2fb4c419354ed2e1618b3cc8dd4ea574ef121b6'},
    'model.language_model.layers.3.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '6be366ef6265c37ce83e084211613628752e2b040c50d2e2b2ffee0c331e2dde', 'stored_values_sha256': '1e0ac9ff532b5384e58e242e7b275f9bc5d7662456687f97d6426a24f752a71b', 'stored_scales_sha256': '04ac2e2d5c6eeae53308f2fe477848b39dc488efa24e56febe78ad37f001ea17'},
    'model.language_model.layers.3.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '4d4d8b8b5d3291b6511fd5da1bef9c28b39e74346f483d910bad0a2f2ebb4234', 'stored_values_sha256': 'eefd22747a51560c8a2ecfd5e1d8e185f8891f551f700a31de82e7bc63423117', 'stored_scales_sha256': 'e2487c3c8896ad63ce42059a0268124f4b55549de955d2c3282e670ecb793ada'},
    'model.language_model.layers.3.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '06e1f3a1a7ec8d074ca4d41be7572a513e87bd543f9bedf9f7925f452f585e00', 'stored_values_sha256': '4cd1911ec45598a56c0efcb6960b1650cfa8765755b70c753deed9308b2f677f', 'stored_scales_sha256': 'ffe12f3cb0c6fc9cd302250e19d688d960bd141af92d4970c33a5f18c275d7aa'},
    'model.language_model.layers.3.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': '3fedff8a1eac07fe4e017967be6d4455eadc97cb09319c2a05ade52204b3ac70', 'stored_values_sha256': 'adcf9c2c56190b8e2097fec8ed8809dc98bf787700a45ff75ac03bfede5bc8cb', 'stored_scales_sha256': 'f3015b4d41f16ae596f1dcacfb3afb82f93e011055eb6c30063227c816b0a09c'},
    'model.language_model.layers.3.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'f587a093830e5e517d82a0704cb89a7423b264579ad5195fa308dbabdbd6753b', 'stored_values_sha256': '13366475f06be402ce1bcb17c6be21bf5b5da69a0a55758331092ece1d6e5c1e', 'stored_scales_sha256': '44ad0c6d82ea201f96a7afa5f755791604dc53ac6eb461b64888d534eb3089e8'},
    'model.language_model.layers.3.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': '6137c4613cb4b3296363b031a67779d5c96f804acb3686664531b6c63044f4f5', 'stored_values_sha256': '041e2d174898e15474d48aeee9bc9b8f84369150d80399343d7a7851b13e33ec', 'stored_scales_sha256': '4c3b25e6c956516fbdb80058c5e708c1a457e93a703d7daf810cf733d20b675d'},
    'model.language_model.layers.3.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': '2e31d5562f86a6bc8347b3074611cd80b83c81ccd798d2b1028db0575aa09045', 'stored_values_sha256': 'f4fbe1d2a67f1673186d197673b88669b3494f3e589143fb8e039e436d2e528c', 'stored_scales_sha256': '5fb1b794debd82fc6d0ab219624335fea3f8ade7c3238dedfcff80a4c60c40e4'},
    'model.language_model.layers.30.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'de72f7631dff5b51ae98007377170aab01ea93d17edcd018303857d94319ac60', 'stored_values_sha256': '75986b0fd5cb294bcd4feccd5cb93ad2462484471bc86394c46bb6ed70499612', 'stored_scales_sha256': 'ea688bf0ef6b8da148117d76616d5bd50e9b094ecab6169be9d74556756ed587'},
    'model.language_model.layers.30.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'fa2cc0c994c673204d5d2531cdd8cc321a6dbcaad7559d28471453f0f64a0788', 'stored_values_sha256': 'c013a46db2d1c90291a975a92017f79d4b803759b510f3dbe02294b5ad36ff8d', 'stored_scales_sha256': '33809c87d65dd2ae545a5dc167327db05e9168ed4c6cbefcca200e3bb6a4cd2a'},
    'model.language_model.layers.30.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'fcb77df634c98055e79cd4bb40650cbb8a58b6c1ef1a26462b758f83418aaec9', 'stored_values_sha256': '882164617c9deb5eb1e709e108d2e28bffc983d089d8fa1ddd79acb6b56e46fa', 'stored_scales_sha256': 'f627d0131008563211cd4bb6e17b75252e6a36d4495d72147f179d4bd431469a'},
    'model.language_model.layers.30.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '27604001b225c8f98f5a9a093712f6a7b0a60c202066c4208bb6aed3666ea43d', 'stored_values_sha256': '1fa7296bc519b0909dfeb55910b76f3b743888e026851543f5dcdfb8036b55ed', 'stored_scales_sha256': '9554e0e722016a5c0e1bc4ba5331e410a968e559d76e26085b9567d158746775'},
    'model.language_model.layers.30.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '5b56883d12c74bb0ab8fee243d0fbc4d9209f61d377d02fe909c64e5d29ab0a5', 'stored_values_sha256': '1b9fc9d6d48faacb638f7faf21ed475fff561646e50eb93a447ce1e629a77487', 'stored_scales_sha256': 'cc617e01c6144e6f823b44c955e6565f50be45bfabf3cba753b348f9ce00dfe4'},
    'model.language_model.layers.30.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '448389750b15e3c015f570ee772ac3eaffc9872a76efd352aefdddfc5a200270', 'stored_values_sha256': '4c4437d6ab3e9842856987b9bb8010c9dd8e0ac0c34bf8e5abeb2a14550dfd41', 'stored_scales_sha256': '87094d77e37618629f7795ce7d9c9e1d53b17aa8ac4e15b93f5813a72dc8c8a8'},
    'model.language_model.layers.31.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '4ced0b87e3d533c8554006fd8e1d7431e08b0b3d94c2379dd3584f3d8553aef9', 'stored_values_sha256': 'e71f1dc9dcc207dc522afda4c46572198032cfe094787995ba7e3e1c9c75c136', 'stored_scales_sha256': '6b79b29cce3fa408bd788dc6d4816deb1da7aa71dde137f5fc9ab9f048e3a589'},
    'model.language_model.layers.31.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '0780377faf6e4e3aa63292db8d6df1c57372758d14d93a6b59943e6f49a6a7fd', 'stored_values_sha256': 'da008f63408a3cd8d4b9fe4de3d5b4b7d320943b44d8b0ba0edaf185063b957b', 'stored_scales_sha256': '7de715c913134a4a7cb343840405308682ea67cac39f8db2e9a94507d7d671d9'},
    'model.language_model.layers.31.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '342d44e2e8808432f87098f65be1d2933d8531ab5e0454e49aac2e4f0af59792', 'stored_values_sha256': '620f73ca9326f34d164fc9500cbe6b00e1bb68b87242d80a131d8ab45515058d', 'stored_scales_sha256': '1549d58754e0881c1acf2270298bdb8471addfc566c70e0bd7e96eea1f89f0d1'},
    'model.language_model.layers.31.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': '4effedee807cfa4776fd977dd0c615fe14cf4a0b195951805cac625ef0c7ba27', 'stored_values_sha256': 'a9a4c0228640965299a4d99b4adec38ccb30902cec96cc9bf1183ea91b622e27', 'stored_scales_sha256': 'f5c56a9ea9370b6f1ac5ef7476e23dd387453e503c038998ec4fa842dc87d78a'},
    'model.language_model.layers.31.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': '5efddd3b95aed9529115e6f01338de97e58b43585992f7f87a237d48a5458da3', 'stored_values_sha256': '601cccc50ea8365d0a44f933a87ef3194afc18c1af83dce7c6c319e07585ad16', 'stored_scales_sha256': 'a4eabb0367ad06787ccd5d40acc095c2eb348ce355fddbbb366e9147c4e500f9'},
    'model.language_model.layers.31.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': 'fdfb53bb58defabafb85814f781ccbcc7a583f23cf16090df32257cc803f1e5e', 'stored_values_sha256': 'ea7f54117614f59216a1498e3be936cae61f847314324e6dcaa3742d70415ed4', 'stored_scales_sha256': '9d3795a149c14da78152f13e2a35895ebe378f5add5d7cb4e1113cfe84d4f949'},
    'model.language_model.layers.31.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': '89dc0f9544284a9ba61f8fb29ca61f35634ae982ef6c25d44da73807cecf4d8f', 'stored_values_sha256': '72f5b0846f197c336b01147c1d0bb58e8450e48f2ab271be847166def463ce7c', 'stored_scales_sha256': 'b6eb4d56a954cc2e13dd89fdc28c24d29430b1a502f21f9ab2f06e24ddd611a7'},
    'model.language_model.layers.4.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': 'dffab8a136414da919146d6d150eac42195de33d7a02815e8d1179d206001935', 'stored_values_sha256': '196b32db5e7bc1d24a2490e154a2ed5545e809354789a8208c59532d165a1796', 'stored_scales_sha256': '6443e145f47414b57a2de541d1c528fd80668482b22c8ed87da49c4b26e6ecd0'},
    'model.language_model.layers.4.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'c4918dfbede343fa351d536e70a39472c106b4f2adf40c74f27228102547b1b8', 'stored_values_sha256': '5cb525d6fedf6efcd53504cb2475f3153eb073b81eb729381b8c21c69d41bb6a', 'stored_scales_sha256': '762872a8a365b037da7ada831a47dc26fce74b9ec187b69b595a7f46598edfa1'},
    'model.language_model.layers.4.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': 'c7b5ae5a7303de4443888498851bdecbc85d4ae5eb09b380b860986136e65784', 'stored_values_sha256': 'ed49d0c647f3f45b7e808f2a6e497aae303ffd11f491f3af59afc12a1578d727', 'stored_scales_sha256': '9ed29c9876871179256aca26e55db8682c52cc6f70b91a6aa8422e6a3ea371df'},
    'model.language_model.layers.4.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '7fa058511d76f6b28788c61cb2383591d6a4f93eaff9b099508a6799218598fe', 'stored_values_sha256': '1395e922e7718755de057749ba05ed76be1ef94040294e3e715aefd9a8abf67c', 'stored_scales_sha256': 'b8dd643f96476dbda9504f2729f3eb61f6a34ac1885cac8483f5b4c1b9b7c468'},
    'model.language_model.layers.4.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '1fb91580b286677eb83fdfd415bcdfdc89c13e7ce1324daa90ad80deb4106654', 'stored_values_sha256': '95ef15ca716b9754f1ff1a156d99c68a6434f5287d2846cff7c91f3e43a72dba', 'stored_scales_sha256': '86fd3ec524c6150baef177fd5eb748649f351ff1521dd927a9355e39f31a95d8'},
    'model.language_model.layers.4.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'a893282e8f6aca629762847140dfe127a5974c2be1f555aee9e722ca55d9e5f4', 'stored_values_sha256': 'b2b22a609eb382b5b75a060cfe910b785ead5aa2b4cfd4d0247e64528ee654a9', 'stored_scales_sha256': '76118755465111395fdcb4a949999b3523b4fcb41675aaa475e7a6e5bd1ea2cc'},
    'model.language_model.layers.5.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '6858975ab1ea99cc8fdf05324b23f33c9da5a9c58de6d1eee9a8da2410585a9b', 'stored_values_sha256': 'd7568ebd005757a9c7ca264badbf0b14b6e9c2675e8d6d67b2683f12d8e5dc5c', 'stored_scales_sha256': 'ea2896b69f2640bbab852376aba2314ccea389065bb26e3f43cd19e09a929358'},
    'model.language_model.layers.5.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '18a60b6b2b81953085c133fd3ae354c3ccbd20207cd3db56f6074ea948f6c040', 'stored_values_sha256': 'a68923956e103387edc71af56f9ba4d748cec0b95fd08392f8b820e41c309a50', 'stored_scales_sha256': 'e99c54bfcfb8730ddc3d8907ec94de0588a99ce52870e2a1f3b01cc8a6c2a1af'},
    'model.language_model.layers.5.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '2be5b88698f4530dc070e57f9b59062aad3d698743a8bbacd8e47996f1a63651', 'stored_values_sha256': '1487ceecc1f5da5d1c2061de5f8689e290e2f580721431f1fbdef37830e46a0d', 'stored_scales_sha256': '45210aa10fb41b0e940914ce0ffff3ad4ce45fe488bfa72cb03f1d9f957cbca6'},
    'model.language_model.layers.5.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '17ef60873469e156eb756c73513b09a44c28f40d0cf4169288b39be0fbb260d0', 'stored_values_sha256': '35509afac2a7075bcc78ab8e43d6b52c003719b10fc5642bc903e72dd0be3f8b', 'stored_scales_sha256': '2e5c46b0cd01836bd60de3825d8acd26a6ea477e4d21b120b4edac74dfe7a31a'},
    'model.language_model.layers.5.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '51e1c23aa080c85493257c6a2eca7ffe41893ed45450ac86f76c7ad37399ee40', 'stored_values_sha256': '44609c1058da1f1b310c6af72d449de3e21743dd450b3286e436e6de5bc573ee', 'stored_scales_sha256': '5dc117710a737c0e018a6b20f8bfb6cd57c8aae5f82c91799b69540022dac206'},
    'model.language_model.layers.5.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'ace52b87069dce79857c6203bb005dc052de40aa478e2f44b8f2cb2572195a27', 'stored_values_sha256': '340b2d54cb16ffd8d34aee862eae7c8bea11693322dc46e8b3369d4c0888224e', 'stored_scales_sha256': 'ea9f705a0d9ef7870b80abdbf3a6463025efb15d35eb9c17ddb1583c488e05f5'},
    'model.language_model.layers.6.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '936cb28dc1e4bb1706975a8e5b6c9abf934cd5431bb72ad49b7ca1051198e1f7', 'stored_values_sha256': '2c7d019a4fea78340076bf1c1d33597b462c6e00f756576dd08eb3686116908f', 'stored_scales_sha256': 'af4d45f12dfdb04a7951cf8bfaed0c8ea7d919cea35c65200721de4507398340'},
    'model.language_model.layers.6.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'fa677f05a5a45fec78f9d7ccba6dc096e31d68f0441602c29bf1a39eafb6ec0c', 'stored_values_sha256': 'adf97ff5797615ab0f292bf408a9418b4aef864b10ea28eaef46ae2648e0cba5', 'stored_scales_sha256': 'f8b140a9d388d675943d3c208da1b039f8499fce888f209ac62d11799dcc90b4'},
    'model.language_model.layers.6.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '905aa0463027a1fac1067611a40dff663dad42394987a2c95f013d8343efe1f4', 'stored_values_sha256': '4436495d5c9cf6b9b6df0bc3055f736ef48e881d4b29e4e0c17fb97d4ce6a45f', 'stored_scales_sha256': '33e9002799d3eba7d291a0239d9c225cdac5822ae0046f99f8b056fb783c76bf'},
    'model.language_model.layers.6.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'fa3280f5a5540ab61f55ff94fc00d757c8c116225002f3eb5f90c1d19afb6bcf', 'stored_values_sha256': 'd6b8658f5a34c045e0470e76f645cb3deb38e2c1913920a3dbfe4cd8812b5964', 'stored_scales_sha256': '4bc81df094e99093d8cb3f7a705815cbc044892cbc17f119b8c1e3d9e2404634'},
    'model.language_model.layers.6.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '9ef966c08a60872e56baf49fb1c05c05daff44165b82f2f061671f464cf6614c', 'stored_values_sha256': '7e2e6a08bfdd223d806b2788addbe6cd23834e85bbcdcd1c62ffa9eb3ba295a7', 'stored_scales_sha256': '46c3bc3172b3e4f1cc01ab27233ff12151d69141038332b22b28b74b03eaa2cb'},
    'model.language_model.layers.6.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '3ea1f76851b0b8917b5f2db6a96900593b9c6ebd1dede17a545c71926a03a1fb', 'stored_values_sha256': 'd266459ba45912832a8e7684df9fe9e3057a2e239b66317b503b6f56efcfbdfb', 'stored_scales_sha256': 'be936dc0ab35ef842bf48257d0d953069adde4c8da79b534e8123fa444cbd4bd'},
    'model.language_model.layers.7.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': 'bcbe9c6acaef856539352b295e9a37cb43a93a69eff591ab476d2a13183ae90c', 'stored_values_sha256': '7ea4bea05a31b492678f0539c54428192eadbcf8a7a35e693782c4f7d318bfeb', 'stored_scales_sha256': '76c2cc77b74a717a6c449632fac5e95b3eeb500ba5b8a38c53f7b4c1dd374805'},
    'model.language_model.layers.7.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '575404964bc9523a00561a04b2cf7cbbdab3c3823d9ae552af775bbf0617ac63', 'stored_values_sha256': '39f5cf49cdf51ecc43523437072f4fe0c64e61237ef9d7530ba513afd8662666', 'stored_scales_sha256': '0f2a7dd68ccf90647a768768358b5e32ae3c07f066fe567a2b491012ce93db84'},
    'model.language_model.layers.7.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'dbd5f4e6c54ed8f7e5b81f50cac14b7171f25e2f53db7559a8986876d990d7a4', 'stored_values_sha256': 'b8c7777efe2c6f806286e9a7cdc108c58bf58a6e75dd9e3a66f900d03d00d084', 'stored_scales_sha256': '8e191f2eb70f210d6e242889551e80a852a34acddd3ec3b4ed2563c9aa770198'},
    'model.language_model.layers.7.self_attn.k_proj.weight': {'shape': [1024, 2560], 'source_sha256': '4a6ab2eade0ecaeed9cedf305ee7cb572b82fd5fc6fa4eb0784fca5a465e84c2', 'stored_values_sha256': '67c2fa389eb88c4c33539348d540f85b20505886d89a647965c12cd14647323e', 'stored_scales_sha256': 'e73cb5b0c51161e8c1c8c93690129409b00de49bd845de43b018ec7b2974fa3a'},
    'model.language_model.layers.7.self_attn.o_proj.weight': {'shape': [2560, 4096], 'source_sha256': '9b4b22b5cbd37f13fce1e87a22c92f387b3dfb81ebebb712ff7c071ec6b6a5de', 'stored_values_sha256': '799fe0957aa7f83951d5570e48c488cfd701ebadc8056ac92c706af6024eece7', 'stored_scales_sha256': '6966ef8bc23db3bf4a4ddb33f6f4343c7befdc36bc61a47df69f3239bdb1bdd4'},
    'model.language_model.layers.7.self_attn.q_proj.weight': {'shape': [8192, 2560], 'source_sha256': 'a58b06968015807ef0462f6c7c22e0c0266f98ffe18ae7a1f31901ab556a2e29', 'stored_values_sha256': 'fc5bdef0ea83c39615297f6d422bd24b092e6e60fb55de8a7418fd03cebbe254', 'stored_scales_sha256': '3c4c331f9960f9299df5eaf2123ddddb5a3f4360cd6adadd48cfbad47db63c93'},
    'model.language_model.layers.7.self_attn.v_proj.weight': {'shape': [1024, 2560], 'source_sha256': '8059c5dc57847de9d256ea470ca34bc821f78c7986985b0b7b95d6d60fa9b42a', 'stored_values_sha256': '8faf40ab5f8d3989de2730b6e464a5c32c12c3f591e41ea2a0028690810a9150', 'stored_scales_sha256': 'fe5650b3c8780511147c7e41050eca5f169f2cf95a681555213af2d5f2da78d6'},
    'model.language_model.layers.8.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '762c39f0080b089d0b54176d622e70b58bdc50feafcbb0fcfc459364c9487881', 'stored_values_sha256': '0827b6d92b46d20e703525c7ca5657be8cdcdd8172e984f06c329239555ee50b', 'stored_scales_sha256': '82496e1f78f3ed23a3dd60da686a7824e5e0515ddb6e2befc2fa6d70064eb924'},
    'model.language_model.layers.8.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': 'd7ae5a8721b57a445366fc9c7a929e27022347e840d376b80c9d555a609ffc31', 'stored_values_sha256': '6afda8721b0e8f052a53a76d102d158233b9dd596319d35259dd81ff3878ae0b', 'stored_scales_sha256': '6e0548164425da282a62dfff0595214577defb444fff2fe0f105df3511c6f396'},
    'model.language_model.layers.8.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '8b38ad7a8ac7d48b666e46fb258f684dcf9205186b56f1e7699b9441fa52010e', 'stored_values_sha256': '08353cbe2f1ca1cc03b32d00798b67b336ab5b532beeef93118078936f5e53ad', 'stored_scales_sha256': 'acb3d84f527966e257161805b445beb28d4e17bdaff3865f822587ccb2dd87c3'},
    'model.language_model.layers.8.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '481375ca06c88dcac7f1b8090cdfd07397c6c1520521e5d45ef1571c73df32a2', 'stored_values_sha256': '689216e5790652abc54fa7fe1ce4202d0c318bf6ce5755a458aa88b2aaa1e5f6', 'stored_scales_sha256': '8fbe862958005c652158848107ba86f84188a3a94c24475aff27bde919f0ed08'},
    'model.language_model.layers.8.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '61eac9eedd5a98dd0e0bf0b1b2ff828b82e00d022b62248aef4bd0535cf3f614', 'stored_values_sha256': '23e1f54f3771fdab71a6a66fc1e47d7b5f489b422f1a95c705427596b81df754', 'stored_scales_sha256': '7f76888382803de1f83ab6ace724377f34b8df66c642d83ca22ab4fd6a20a103'},
    'model.language_model.layers.8.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': '2e77547cb5697f7575840687250cd5e4833272b684ed6c586918feeaf3bdd1bc', 'stored_values_sha256': '5cb1cf73be2b4199025fed4f09a2d3d2d61c457be4887e875ab8509eed931f46', 'stored_scales_sha256': '9f6d42c665836978a0643afb6bcb4ce80e147011f13a46d57f8bbfa3d984a5a2'},
    'model.language_model.layers.9.linear_attn.in_proj_qkv.weight': {'shape': [8192, 2560], 'source_sha256': '97a0747207190ba5037756999fbd98f2e33507d5d3fc67d360a68154846db190', 'stored_values_sha256': '7e5a8a48da1d49b2a8e8cfc02e68285d6ab3d868e3842db7cf54d97f3b07744c', 'stored_scales_sha256': '9ad7126faa31d9f9bd031653803046d24e00257fb7e5a1627b529fdd1b302c60'},
    'model.language_model.layers.9.linear_attn.in_proj_z.weight': {'shape': [4096, 2560], 'source_sha256': '3b327952a53c7af71f50a207b61df656cedb214d43fabaf2b39e5093604e5790', 'stored_values_sha256': 'c6cbf9fbd6f3cc78672a7af6bb01824297d17d329b6460688fbc08453bdf5b37', 'stored_scales_sha256': 'd6da7ac279d9d9eed771d12545f2e2034792cad1a0059b2c284da8aabba42b52'},
    'model.language_model.layers.9.linear_attn.out_proj.weight': {'shape': [2560, 4096], 'source_sha256': '1e7a6f7ebec3dd7063500433bec0c87fc80a98aba594db7bd44635d2acc3ac04', 'stored_values_sha256': '9c10f31261832b5b92762e88916b6cdec1ccf4e4d94991ab3cc4312dc716fb39', 'stored_scales_sha256': 'b9fd2164d426184f200af213c73928f8b5bef829bd5a64eb3d33d6f0d03f0364'},
    'model.language_model.layers.9.mlp.down_proj.weight': {'shape': [2560, 9216], 'source_sha256': '4ef4416b94fbaaa82709c9d03fa27497cf3e5081789dd5a4a4aeeb225808895b', 'stored_values_sha256': 'cb61ca2fa9d9f9dabef825ae6dd88b8b2e44b8699e024c74773536afebeff13e', 'stored_scales_sha256': 'd65d3f4b6f4390787c5947e9eb5ebe809610fb801e034558582eeb4ede4a9a92'},
    'model.language_model.layers.9.mlp.gate_proj.weight': {'shape': [9216, 2560], 'source_sha256': '22a1c46509f5e4d4b6a08df28cecbedc2046741009fb8cc3324cbe2aaeb850e7', 'stored_values_sha256': 'ee07d2eda5e635c03dc8b18430aa516291c950a69a83474f61fe44af29624d3e', 'stored_scales_sha256': '85abd96e527ea8bc47e9156f7903a2e24bdd34e2e7189d2166c80f80c1d94644'},
    'model.language_model.layers.9.mlp.up_proj.weight': {'shape': [9216, 2560], 'source_sha256': 'af0f5ee2f0880dde49bc5d5727d834ab391b48abacea41382e0a7fd9a0b9bf68', 'stored_values_sha256': '205fa401d229e885c642af25f6ce41306f0754b55814a51ec11902fae49b260b', 'stored_scales_sha256': '4af4cdb069d13622f057eaa376dbec63073f5b82635c7d6cd0672c0ef80671b6'},
}

RETAINED = {
    'lm_head.weight': {'shape': [248320, 2560], 'dtype': 'BF16', 'sha256': '2a68153b498532801ab605bb03fe617c57b3e6a3ba019301fb70ada0c864a2f7', 'source_key': 'model.language_model.embed_tokens.weight', 'source_dtype': 'BF16', 'source_sha256': '2a68153b498532801ab605bb03fe617c57b3e6a3ba019301fb70ada0c864a2f7'},
    'model.language_model.embed_tokens.weight': {'shape': [248320, 2560], 'dtype': 'BF16', 'sha256': '2a68153b498532801ab605bb03fe617c57b3e6a3ba019301fb70ada0c864a2f7', 'source_key': 'model.language_model.embed_tokens.weight', 'source_dtype': 'BF16', 'source_sha256': '2a68153b498532801ab605bb03fe617c57b3e6a3ba019301fb70ada0c864a2f7'},
    'model.language_model.layers.0.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '087f17ffbaca175b64d5a00aeab2cbc3aebbaf8ffb766c53ec93297904b274d1', 'source_key': 'model.language_model.layers.0.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '8104f6b0c777fd9bc60925f81a7179cfb7bf9621b4abf26a4d0f98b6e9a9bfe9'},
    'model.language_model.layers.0.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'a96c799c5535baeb444d5440ab8a7bc0ba297581f6dcf45883085d84935fcb1a', 'source_key': 'model.language_model.layers.0.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '6430c4f37da24f2359ac66cfdf295fbaf09f6e24e8ece890eb98c441ac6d0523'},
    'model.language_model.layers.0.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'f8e622f2787b8bb6fabf020039d14608cef3b98eb7aa7ceec782ccaac71f4242', 'source_key': 'model.language_model.layers.0.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'f8e622f2787b8bb6fabf020039d14608cef3b98eb7aa7ceec782ccaac71f4242'},
    'model.language_model.layers.0.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '9862c6e16cc1adef8de148c5ac316bb8323db3f956f99c131cb167fdeba25d15', 'source_key': 'model.language_model.layers.0.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '9862c6e16cc1adef8de148c5ac316bb8323db3f956f99c131cb167fdeba25d15'},
    'model.language_model.layers.0.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '236b623c83199accbf2552c1b5519b8d977fe1813fcf429bce80f84d66e8da78', 'source_key': 'model.language_model.layers.0.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '236b623c83199accbf2552c1b5519b8d977fe1813fcf429bce80f84d66e8da78'},
    'model.language_model.layers.0.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'c886de583bf6553a8d814801687cd753e6ee6440b1f2788ddb75149104183834', 'source_key': 'model.language_model.layers.0.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'c886de583bf6553a8d814801687cd753e6ee6440b1f2788ddb75149104183834'},
    'model.language_model.layers.0.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': 'f0da528c31ca70025d0aea92f9f833cb4cf803021406ffd8311d8b4d3804de69', 'source_key': 'model.language_model.layers.0.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'fecbf8328e83d0ed32ba85dcc096a922381da16fd22e430752a336bae198b701'},
    'model.language_model.layers.0.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a943da9af997ce5d40a5482331f322584f66f3a3c18f1c519d0cb9c26378c2d0', 'source_key': 'model.language_model.layers.0.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '6788792cb525e77512a4b5aae181bd42bf69fe1a3e983683b22f04fec37f947e'},
    'model.language_model.layers.1.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '4c009cd49c15fd560974299d1e5752b11cf166dd7d885461a0684b78638dfc28', 'source_key': 'model.language_model.layers.1.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '7d29ab3543ddeb306f987842e00ee08d6e956f090a158ab0f926c2a36e1c41f7'},
    'model.language_model.layers.1.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'd9891ec375fd91f24bd0fece35dd5b6335cac43d67542fb3ed7b6488b39c3ff5', 'source_key': 'model.language_model.layers.1.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '7ff5540a8ccdb43c3b0ac3848841891e3a5ed231875ce413096169ccb98340aa'},
    'model.language_model.layers.1.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '3b99cdc7e4edab9c266416c948a284bfd51cb4035a83d09e67f8b6a7f76e4cd6', 'source_key': 'model.language_model.layers.1.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '3b99cdc7e4edab9c266416c948a284bfd51cb4035a83d09e67f8b6a7f76e4cd6'},
    'model.language_model.layers.1.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'c185e125bf484548761fcef22902961616fa8f6dd46ef9aea2ae1f0f9556adac', 'source_key': 'model.language_model.layers.1.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'c185e125bf484548761fcef22902961616fa8f6dd46ef9aea2ae1f0f9556adac'},
    'model.language_model.layers.1.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '15b10417d5520ea356f08da942e968dcce8a51255c46ba72d9a558f5b2bc054c', 'source_key': 'model.language_model.layers.1.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '15b10417d5520ea356f08da942e968dcce8a51255c46ba72d9a558f5b2bc054c'},
    'model.language_model.layers.1.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'a736e33c2b507bbc93b14df02076cf342a76b37ae79390518061ed0db2de7d92', 'source_key': 'model.language_model.layers.1.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'a736e33c2b507bbc93b14df02076cf342a76b37ae79390518061ed0db2de7d92'},
    'model.language_model.layers.1.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '8d356e686c96ece3b3a2fef57dc5c6b1cf87da5e83f605efab47a7df663afed8', 'source_key': 'model.language_model.layers.1.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '2a179e60b1d815e47ef12dabdf288035441c10a768e8649561c5f1bcaf30b957'},
    'model.language_model.layers.1.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'c702b8ca9ebc43043f13dd7252f52df3f08aaaa44c2a037a165f3bc91caf786b', 'source_key': 'model.language_model.layers.1.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '3f8bf985cd77f6ab425e525a60a788ddf04cf0c806cd7480c0d6d34164dbfec0'},
    'model.language_model.layers.10.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '5b7a58d31d260f18e63617c3a8b7352bf3c7b9a5b263e5a77a49bdebb2aaa7c8', 'source_key': 'model.language_model.layers.10.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '0321b0b0d5c47837f1c146a34532ef53029044e62281d0478fefb3d676d6417f'},
    'model.language_model.layers.10.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '90c78bd2e23dd8ecb5030e8dacd973a2352dd48f7df193e4d0abeb5d5d5d99eb', 'source_key': 'model.language_model.layers.10.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '0fd6e9d70525938e98849de05d4b500e46e0f503380d0b2c82344685282b6708'},
    'model.language_model.layers.10.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'e5a6b4f89460a3fc5f3acce8ceadc19cb4bb8e3434e909a8afc9c0c29330a6d9', 'source_key': 'model.language_model.layers.10.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'e5a6b4f89460a3fc5f3acce8ceadc19cb4bb8e3434e909a8afc9c0c29330a6d9'},
    'model.language_model.layers.10.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '124d1d9e7f8d0a59e83414af7fd0e1204fdad14b45c154a51955d775331f1d50', 'source_key': 'model.language_model.layers.10.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '124d1d9e7f8d0a59e83414af7fd0e1204fdad14b45c154a51955d775331f1d50'},
    'model.language_model.layers.10.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'e3fe3d779decfa82e8d0a33d3f04bb4354fc92efcf40dcd3092b598c8a31617e', 'source_key': 'model.language_model.layers.10.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'e3fe3d779decfa82e8d0a33d3f04bb4354fc92efcf40dcd3092b598c8a31617e'},
    'model.language_model.layers.10.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '7b9a70d448e4c45cb40ebb184b677d511afb18b3126daaae66af37883b87533a', 'source_key': 'model.language_model.layers.10.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '7b9a70d448e4c45cb40ebb184b677d511afb18b3126daaae66af37883b87533a'},
    'model.language_model.layers.10.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '115614955fa6ee3b218b12bc60070c56090941a79618b0af83fcf99d5d5eaf02', 'source_key': 'model.language_model.layers.10.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '9cc8ffaf05d629c0c8104adac39acbfb382cbf03d00c61b1c602c8620d8d71e3'},
    'model.language_model.layers.10.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '7343b46eedcc8e497fdb8cb383647577a6896b952f5d81e93daaf1b27654cee3', 'source_key': 'model.language_model.layers.10.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '7433b0fd17b4eeda5f5b88fc6aa3604ba37efb81d10786624eab82a90521aadf'},
    'model.language_model.layers.11.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '5d6245d09ce0bfc1b311dd29593b29ef64d1f907644ded80172bb43297323f45', 'source_key': 'model.language_model.layers.11.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '0b3319102c276403966d93b12329b04b70419f064ebf6bb4f2622c574d29dfdb'},
    'model.language_model.layers.11.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '96ce63b211bd556756b3e5669ea2b49ca9d0cad4eb1d93a7c4826000812ef5fe', 'source_key': 'model.language_model.layers.11.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '3087e75ad3c2f6c040bee639ee83573df3fde2accfd5a197e34bb15eaa402532'},
    'model.language_model.layers.11.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': '08fceb159b4ec2de07d52eea74c46ff83f7850a88d4f78f4700100c17434156e', 'source_key': 'model.language_model.layers.11.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': 'd48ea812beb21d6b4ff5ea52bb336b8ac009c0df82fd541ae4543bb8094c7f89'},
    'model.language_model.layers.11.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': '41aeb5998c49eb891025fce574573fb6ac8102d090117200168b6b7f63dd723f', 'source_key': 'model.language_model.layers.11.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '689538b5568dcef8873467420336d9462c3690db474a531c0d862b16eee59882'},
    'model.language_model.layers.12.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '8df0e4133904d117b2c0ed08eccf0bee101db5dc33981fc14f8ecfe4b9a7dbf0', 'source_key': 'model.language_model.layers.12.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '9d2752ba047afd551c54de7eb68ed136b1235706a0fe6d9ed824897c90a4487a'},
    'model.language_model.layers.12.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'ab5e4c892f736c2c1bca8e0953f40dd3f44d6db564c4b7d4d17020a8d55fd7fe', 'source_key': 'model.language_model.layers.12.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '70e702bfaefec7110bb03596a10f1ac2512b85acb9e0d4395f06c58f68e2580b'},
    'model.language_model.layers.12.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '58c3eef13aff325359e4dc255ac6de2f29342678659ece81a099eda3f38c7cdd', 'source_key': 'model.language_model.layers.12.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '58c3eef13aff325359e4dc255ac6de2f29342678659ece81a099eda3f38c7cdd'},
    'model.language_model.layers.12.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'da250802c105834611ce55a986300f292282956910e93c3ef39ee108564ee5dd', 'source_key': 'model.language_model.layers.12.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'da250802c105834611ce55a986300f292282956910e93c3ef39ee108564ee5dd'},
    'model.language_model.layers.12.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'e385161933256ac1dbb22844e160fdaf495ada12864e5007c13eb1887666908a', 'source_key': 'model.language_model.layers.12.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'e385161933256ac1dbb22844e160fdaf495ada12864e5007c13eb1887666908a'},
    'model.language_model.layers.12.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '1cd1de63edae4fb8b0f6440b31b585c4c240fb996e51ba0e4de087636335f5c2', 'source_key': 'model.language_model.layers.12.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '1cd1de63edae4fb8b0f6440b31b585c4c240fb996e51ba0e4de087636335f5c2'},
    'model.language_model.layers.12.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': 'd43d4370f96bc95af5841fdf38220f3e98f6095862d121b063d52ffd4295135a', 'source_key': 'model.language_model.layers.12.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'ada83d241154af2c49b90396e8ac3d34b7b2314786c825a297f599fe3af0ada6'},
    'model.language_model.layers.12.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'ced719bc3be3e86399eb94fa50d384e944b7070c826236630eea981eb5ff3176', 'source_key': 'model.language_model.layers.12.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '454c8d3e85c88b98179e9d6142113f5c7c1170df77d220d02359c286e89856b5'},
    'model.language_model.layers.13.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '5ecfcfad9c55c5886248c5296382593505ecdbddb5e9fae9f658e4e6a5ca82b2', 'source_key': 'model.language_model.layers.13.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '498336e0270b512c0646a4bb12a738a4314f169780c16888d913f8dae4863a57'},
    'model.language_model.layers.13.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '336a27e1204580bbe14cc538421ee87473f451e8312ab6a0b2629d0db10d9262', 'source_key': 'model.language_model.layers.13.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '6f90237b4eec857dd3e051be3e0b0382b8ac7e78869e8ddd6a08e3311965dfee'},
    'model.language_model.layers.13.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '92a2dde264000cdfc8c724d0309c7fe06644c6782abc695b3416d4c47addcfcb', 'source_key': 'model.language_model.layers.13.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '92a2dde264000cdfc8c724d0309c7fe06644c6782abc695b3416d4c47addcfcb'},
    'model.language_model.layers.13.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '8bd6fc07a887e0fa3292ac279ecedcc3902dbac4a05b024f5dcb14c3788f5f16', 'source_key': 'model.language_model.layers.13.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '8bd6fc07a887e0fa3292ac279ecedcc3902dbac4a05b024f5dcb14c3788f5f16'},
    'model.language_model.layers.13.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'bf94f179c7e914dcf1a4a0ad1e537b1ce347f2da0a68bc2cd14d03e77f47a8e4', 'source_key': 'model.language_model.layers.13.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'bf94f179c7e914dcf1a4a0ad1e537b1ce347f2da0a68bc2cd14d03e77f47a8e4'},
    'model.language_model.layers.13.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'da98fab62a7d79c43522e7dbb76b5ea255ae5c573af68185e5e915381127c437', 'source_key': 'model.language_model.layers.13.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'da98fab62a7d79c43522e7dbb76b5ea255ae5c573af68185e5e915381127c437'},
    'model.language_model.layers.13.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '80751b690c142236548d7a5eeaa8c6636569d2608ff2c87c3b4270c8e0298b46', 'source_key': 'model.language_model.layers.13.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'c3f804afdfd9abaeec60da5ac4733a1a342679201b064d2b159fee350c934067'},
    'model.language_model.layers.13.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '073efdb29d65f526da98334d7aaca12d38b50fd7e72a0e0c4ece343184aea28a', 'source_key': 'model.language_model.layers.13.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '917748a0efd2d468953c6f27761ff81f39d7d7e114548ceb6e75c65250e6a851'},
    'model.language_model.layers.14.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'b405fc3ba1593a0df0e98ef7398ab3d8f94ecfc5a3d81b0930ecc813d96d3609', 'source_key': 'model.language_model.layers.14.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '8acf04e662fc775dd4a09836c611f1d2d8e3b4aa84cbbe27527e801f589fbeb9'},
    'model.language_model.layers.14.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '1fa08263299a82fd3a5aca7555005d699c3054922cc32eada2aa6589eff14286', 'source_key': 'model.language_model.layers.14.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': 'd33c419b3d4b915d32ac88bad1cb4f24e81a02f7d6430a1f9b02d5f3a3fdbb63'},
    'model.language_model.layers.14.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'c87a3f4380849ed8522740dfc0f181ad7a87955aed00908604e2f17458d02a8a', 'source_key': 'model.language_model.layers.14.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'c87a3f4380849ed8522740dfc0f181ad7a87955aed00908604e2f17458d02a8a'},
    'model.language_model.layers.14.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '571f7044c180d0229ad79ba3203ac0d91bb4d02eed7453d256f18029d02454f1', 'source_key': 'model.language_model.layers.14.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '571f7044c180d0229ad79ba3203ac0d91bb4d02eed7453d256f18029d02454f1'},
    'model.language_model.layers.14.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'ee16b363287f8d3cb2d976b515239c19687816a6eae669e43c542c85c2f6be16', 'source_key': 'model.language_model.layers.14.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'ee16b363287f8d3cb2d976b515239c19687816a6eae669e43c542c85c2f6be16'},
    'model.language_model.layers.14.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '759479747b35808dd3817358e367726034a06fb88be463fe82523da2ad83051b', 'source_key': 'model.language_model.layers.14.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '759479747b35808dd3817358e367726034a06fb88be463fe82523da2ad83051b'},
    'model.language_model.layers.14.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': 'a0e1c51659e42daf588c6a30198377f0970d90a660c5ac06d9f3dba75b5d656f', 'source_key': 'model.language_model.layers.14.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'c810f585f1af5e8cdf94c6a76054ab892549c53276f811812352d4d17e712dd0'},
    'model.language_model.layers.14.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '6d04c68e32678da51b3d65814a66a03b3575161cc344007302c9f7d630fcba2b', 'source_key': 'model.language_model.layers.14.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'b2e07d2a1bde0f05e070285e72f5ef537dccf4fb3cef0296e4dc71cf0c9c2039'},
    'model.language_model.layers.15.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'bd9c7e51d7149b846ba59013cc182c98e7f6cf15a8f7613b3e980b80bf3cbab3', 'source_key': 'model.language_model.layers.15.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '693c9409a73005cd819168bbf2e862965ca997a933e94475a8c5444e05855774'},
    'model.language_model.layers.15.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'aaa05e12496087f66febcf658f60d120bf6f61aae455267d8382d8caffbfa1c7', 'source_key': 'model.language_model.layers.15.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'd0adbd7f50153fc582495ea479604b4d6f609b29e36bdedeb368cb2694b3e97a'},
    'model.language_model.layers.15.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'f34307bae52a119f69521c049dd8c86359ab05e59a2c07f283cf0ced88b43bdb', 'source_key': 'model.language_model.layers.15.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '8eb83402645c7786fe9762b39161bc342312893cbb77bc8f514c5a4170d3ede8'},
    'model.language_model.layers.15.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'e280423c90ab8fd9deed16dfd382376efad2d22ff5e3c29fc583617541227872', 'source_key': 'model.language_model.layers.15.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '1b5cc613c1755ff97fea358b4f1f691cbe8df3333e1555098f51e377c1bbc0dc'},
    'model.language_model.layers.16.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'fe4f09c6bee3a56227caab671422e7a093642fdbc93666df61454cc03f97af38', 'source_key': 'model.language_model.layers.16.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '98a7fec0d142f1c5138f0df4cfd2f5dff7203dc7a66ba5e39d6d5c05803de26a'},
    'model.language_model.layers.16.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'e9546c185e23d806a11bda49bb522d7a60020da8169a1abac9b1579823b8bc07', 'source_key': 'model.language_model.layers.16.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '4b32f71370c8f314fd0838f7a12c86ddd03b01b58cade98ebd422f037e9ee340'},
    'model.language_model.layers.16.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'e30131083c7cf13570dc499e19cec0d15ec7aca852b118e410a97fbc835d5b47', 'source_key': 'model.language_model.layers.16.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'e30131083c7cf13570dc499e19cec0d15ec7aca852b118e410a97fbc835d5b47'},
    'model.language_model.layers.16.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'd48393e930771b93a5bc2be684f92eb02c9a4d988e4c1d7ea1d3d8ad01873735', 'source_key': 'model.language_model.layers.16.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'd48393e930771b93a5bc2be684f92eb02c9a4d988e4c1d7ea1d3d8ad01873735'},
    'model.language_model.layers.16.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'a2b3a359ceef642f0646081ebad94c03cde1bf8f5992705f2cf684f16753932a', 'source_key': 'model.language_model.layers.16.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'a2b3a359ceef642f0646081ebad94c03cde1bf8f5992705f2cf684f16753932a'},
    'model.language_model.layers.16.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '79345b534835e37f17421473840783ba991270717baa1bf339bd818d1b4c189c', 'source_key': 'model.language_model.layers.16.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '79345b534835e37f17421473840783ba991270717baa1bf339bd818d1b4c189c'},
    'model.language_model.layers.16.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '9165c2d4e5f0793edd253d7d8747ac77e08995868fb68e2587579c43c6d18f81', 'source_key': 'model.language_model.layers.16.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'b39a5d47ba7c1b0c1e91526ac6f91b59d70bb9a42e024113bd59dd721455dc59'},
    'model.language_model.layers.16.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '862dad53c7bcd0249b6dec7178b4d62ea28f81eff7868ca699c4c92b96d63843', 'source_key': 'model.language_model.layers.16.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'd024668922e45a06d11a0a1105881a0c5ccd3c506d871d0d99e60a80352867dc'},
    'model.language_model.layers.17.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'b5dac9f8d11fc4a5b55c6bbb13601adda6d8e65090bedba5e1427e2d86a62543', 'source_key': 'model.language_model.layers.17.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '155987686f89474f3a824d0bd917e2ca1c64df205c7f01b9223629085baf9014'},
    'model.language_model.layers.17.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '031d60f629244aef4059db01b28eccd69b5eb4c3901541445919a48593f98a0b', 'source_key': 'model.language_model.layers.17.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '38c11d9929c55f07889e4ea10f7622296479c0c10ed3d0f2e376219157692ff7'},
    'model.language_model.layers.17.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '70b125e18281e34e164b94c03dceb233f516d79e3136419d03ff3bbd7b149a01', 'source_key': 'model.language_model.layers.17.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '70b125e18281e34e164b94c03dceb233f516d79e3136419d03ff3bbd7b149a01'},
    'model.language_model.layers.17.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '622d552c343bea90c87df5f2b89781387bbcc29d64cfa08020fb128b01a49cc6', 'source_key': 'model.language_model.layers.17.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '622d552c343bea90c87df5f2b89781387bbcc29d64cfa08020fb128b01a49cc6'},
    'model.language_model.layers.17.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '9ae252465f2193cc736bf896d94bf08dec40b249ed34b542fba3ed1c9067eb64', 'source_key': 'model.language_model.layers.17.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '9ae252465f2193cc736bf896d94bf08dec40b249ed34b542fba3ed1c9067eb64'},
    'model.language_model.layers.17.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'e6d213e53e75098dda9b366a657ebea762f3275cc25fb0d5e8968e10f634b61b', 'source_key': 'model.language_model.layers.17.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'e6d213e53e75098dda9b366a657ebea762f3275cc25fb0d5e8968e10f634b61b'},
    'model.language_model.layers.17.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '1059fc55d601a9721a9610e2de40ef9c7a041e9505485e286d94596a7069f21f', 'source_key': 'model.language_model.layers.17.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '2a85a07045f8279663427d8b6e510477a01695c93454999636965e529e001a37'},
    'model.language_model.layers.17.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '690a3eec2f24ca3429dfc894be4f10756b65d5933a4650d034b0a1b021261bf7', 'source_key': 'model.language_model.layers.17.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'c6dad80df7eddd76bebc8ca158d6ca191595c493ade5e1d084865fdf57769d9f'},
    'model.language_model.layers.18.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a6d702ed32668e67ce3446a7fb14584b2db85a3b0be03f200d6ba4f2ef5f8b59', 'source_key': 'model.language_model.layers.18.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '7809efc7923abd21932c53700fe3097796395e5a6f72796357322c75f5d05f13'},
    'model.language_model.layers.18.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '63672e0c6bbd0f60e02dedbff432b05d2a388b26609ac24d70bc177082403286', 'source_key': 'model.language_model.layers.18.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': 'aea4d5052df09846acf97654beb4104ae6ccf7815bd2395c3861a239fe4359af'},
    'model.language_model.layers.18.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '87251da8cf0ede7a1fe2511e71d2430f3d15032f25615a00b67d952e88deb87b', 'source_key': 'model.language_model.layers.18.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '87251da8cf0ede7a1fe2511e71d2430f3d15032f25615a00b67d952e88deb87b'},
    'model.language_model.layers.18.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '97eac215b77c5dcba4e5ccab9a489bad719b0f1407cd63d47622181cb9fe35f2', 'source_key': 'model.language_model.layers.18.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '97eac215b77c5dcba4e5ccab9a489bad719b0f1407cd63d47622181cb9fe35f2'},
    'model.language_model.layers.18.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'def9d5f18a074fa52767d159f49c48f15413aa0b3aa7b45ee2ff8c56f82aa0f2', 'source_key': 'model.language_model.layers.18.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'def9d5f18a074fa52767d159f49c48f15413aa0b3aa7b45ee2ff8c56f82aa0f2'},
    'model.language_model.layers.18.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '6efe19a079b026b4d67dd40d7075047281d9a8b73a9c49f71d512171e7f150c6', 'source_key': 'model.language_model.layers.18.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '6efe19a079b026b4d67dd40d7075047281d9a8b73a9c49f71d512171e7f150c6'},
    'model.language_model.layers.18.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '5fe96de3937406735acbb23b6556c7939d3be97a8df04e208fe1aba587d24551', 'source_key': 'model.language_model.layers.18.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '8ba552913be91bd12cec92ef6e6f4d00901d42d2bf391a3e308b53df288f8fe7'},
    'model.language_model.layers.18.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '368d832bbde1e3dc9d43b72103573358bc152c98d5421020681bf1db283a68b3', 'source_key': 'model.language_model.layers.18.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '6e90c21b436420932bcf6d2bd18d18dbeb40ed5bdea50fbeb0f5f6766b17228a'},
    'model.language_model.layers.19.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a0ccbd75ca222c0bf334ee2b714cfad71c675bc528ed01af2f6143f73fd9069b', 'source_key': 'model.language_model.layers.19.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'b76ca3d97ea27390636c83f762bc8fd76ecf43dfa2c88d66a2781a7b8edd82e2'},
    'model.language_model.layers.19.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'd7f78a11d958f71d71007e069f77075ac976ccadae9feda9629394b400204ff2', 'source_key': 'model.language_model.layers.19.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '74ed4d58bcb74034107add406a4de8c6408ac4f7f4f8a49fa4d52f5279080e3e'},
    'model.language_model.layers.19.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'e88b3e351f0a157de6bd2705f9665c5b49916c8bfa57a375b65629e4285465c6', 'source_key': 'model.language_model.layers.19.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '97308c124435ad0b8a57952426e3bae04d618b63a012b1097a6c9fb183b41471'},
    'model.language_model.layers.19.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'ca477b80adc47751fba535e3a53b2209ab50b1442c982c565c670be9df31b555', 'source_key': 'model.language_model.layers.19.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': 'ee66b46c11bba636c2e3be268d5c5b721448bf8d00d6b77483c61a6670eb7341'},
    'model.language_model.layers.2.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'dffc3d46ffa21b490960c1666a454f79362ab3f5391c73b38fc0bfd3080a2234', 'source_key': 'model.language_model.layers.2.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '89e5804d2fdbfa91ca97628341231a0b2dbb26ca19ae88ba8f43e2383ed1b5b7'},
    'model.language_model.layers.2.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'c5d50d57b197d8473635e71ec1a53f41e8933d4bce4ae6cf84862ec3795960b4', 'source_key': 'model.language_model.layers.2.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '3ca2471722e54b46b6a283b391311cd9ea75fe0bc42b7b4815864e6ef1abce29'},
    'model.language_model.layers.2.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'eb804af2bf26d1748c8778b0096d96bf25c72d4ef7498e03bb38c80eaf55d936', 'source_key': 'model.language_model.layers.2.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'eb804af2bf26d1748c8778b0096d96bf25c72d4ef7498e03bb38c80eaf55d936'},
    'model.language_model.layers.2.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'c6d33d7d1831bdb81f013d7f5a9e5a43148000a6c09fb88e638eab297b036153', 'source_key': 'model.language_model.layers.2.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'c6d33d7d1831bdb81f013d7f5a9e5a43148000a6c09fb88e638eab297b036153'},
    'model.language_model.layers.2.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '55693f7c8321aa145ab04670a9c48d0af31f1e631acfed7d9c5cb0ba884d0197', 'source_key': 'model.language_model.layers.2.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '55693f7c8321aa145ab04670a9c48d0af31f1e631acfed7d9c5cb0ba884d0197'},
    'model.language_model.layers.2.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '91410c0f6eb718dca353cd02125090994e59757999d4db1591d5362ff4ec2e4a', 'source_key': 'model.language_model.layers.2.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '91410c0f6eb718dca353cd02125090994e59757999d4db1591d5362ff4ec2e4a'},
    'model.language_model.layers.2.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '003cbbd621d086720c97487a68b6b9098fc1da3d9cbe4e3b97297bb48602bdbe', 'source_key': 'model.language_model.layers.2.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '3f90759298844b0fd084f9121b4efbcba4fc0bc83eb7bba056219c73339e8d88'},
    'model.language_model.layers.2.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '07e2013bb51b2a642f2193b3b05d253656a15c3b5181d32dd1dd6fd4a7eda78f', 'source_key': 'model.language_model.layers.2.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '2675e44a854dae20c0ed120d0f11e445b352c4f7adbcee9c4290c3270b1d0e66'},
    'model.language_model.layers.20.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '32a8cb5a48b3aa319c7f548ee4b38d742d3c089e754d195f1e8091f23bfaf40d', 'source_key': 'model.language_model.layers.20.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '4d5ac5d6935dc0cc2f21d3788f6fc626209f0cb0db0e0c45076753f52bcaf3f5'},
    'model.language_model.layers.20.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '97a38de7617bd94acb4f1f2b40b3130894841396fcc5b6fce067468ca9c92e1c', 'source_key': 'model.language_model.layers.20.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': 'd535bc8b556f0136e796f2cd53ba9e8e385aba899e8d6eea57320c5cd0cc6565'},
    'model.language_model.layers.20.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '0f03f635ab5f0f27d5d1be569a2aa1b2968e8e011255d3057259b3ad3b816e19', 'source_key': 'model.language_model.layers.20.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '0f03f635ab5f0f27d5d1be569a2aa1b2968e8e011255d3057259b3ad3b816e19'},
    'model.language_model.layers.20.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '6201ccbd7846ad8724736006528b02df6027355fb6765ab91dae22df34b8546a', 'source_key': 'model.language_model.layers.20.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '6201ccbd7846ad8724736006528b02df6027355fb6765ab91dae22df34b8546a'},
    'model.language_model.layers.20.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'c261ecb3a7c3e5278f0b3ea34711a675fc937b9a0047fc6306b37c4f59789b59', 'source_key': 'model.language_model.layers.20.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'c261ecb3a7c3e5278f0b3ea34711a675fc937b9a0047fc6306b37c4f59789b59'},
    'model.language_model.layers.20.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'f750c9b1a0c3bc0706303b9c3e145ea50e0bb2a05e0172721a641950e188b165', 'source_key': 'model.language_model.layers.20.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'f750c9b1a0c3bc0706303b9c3e145ea50e0bb2a05e0172721a641950e188b165'},
    'model.language_model.layers.20.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '16e84a31ec659b1873b6fd237dcf00001299d9c119f05a2d64333b7095a34fa4', 'source_key': 'model.language_model.layers.20.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'd89e3497da61c4ac36891a6fde6e907e124d2429a3def29daf1814f73e468766'},
    'model.language_model.layers.20.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'c5eb678c917aec8b8870be8400d1511be4586eb2f2d158860c76923ef85cbd10', 'source_key': 'model.language_model.layers.20.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '9da99bca0d7b239c3f972413f2e21512d61a39baa017f447da9ef9a66322bcee'},
    'model.language_model.layers.21.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '621e108b0006e339d113428ad97dfba4f412a90ad24004ac97a0c77f40fd63cb', 'source_key': 'model.language_model.layers.21.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'fab45654741df2b34385fb12fbf9604800c0ce86d1859e38b48a8773549e5f4c'},
    'model.language_model.layers.21.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'd3c3475278fae283a3599bf70de7f316b7d78f21154e4bcdf84816a3db812037', 'source_key': 'model.language_model.layers.21.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '46b15203078884048550deddeb7fff61776007c491eb66fd47c2aa23842d2c9e'},
    'model.language_model.layers.21.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '1cb90c36b1791d779a369537882ddb500bf3289089bdc9e4274f7aaf5bf35123', 'source_key': 'model.language_model.layers.21.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '1cb90c36b1791d779a369537882ddb500bf3289089bdc9e4274f7aaf5bf35123'},
    'model.language_model.layers.21.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '00e7306f72d41aa8308dd0932f5e82561944f1856cf637e0a52ae32e501097ea', 'source_key': 'model.language_model.layers.21.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '00e7306f72d41aa8308dd0932f5e82561944f1856cf637e0a52ae32e501097ea'},
    'model.language_model.layers.21.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'd9f7215b13212f146b104539288d98e9ee39670127b68d0ae367981a7e47ba32', 'source_key': 'model.language_model.layers.21.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'd9f7215b13212f146b104539288d98e9ee39670127b68d0ae367981a7e47ba32'},
    'model.language_model.layers.21.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '1dd2187f7e4a90f4c12f152640fb9d323cabeea4f4b0e387204c419b269034d4', 'source_key': 'model.language_model.layers.21.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '1dd2187f7e4a90f4c12f152640fb9d323cabeea4f4b0e387204c419b269034d4'},
    'model.language_model.layers.21.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '7842aa76082de466d923dad5574933902be760d47d68c115a92b86203c70d5ed', 'source_key': 'model.language_model.layers.21.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '850d97eac054888d697f08fc26dc53564485cc7b4ec0a407adb7663aacb54249'},
    'model.language_model.layers.21.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'f30a57921a2b4bca56e006798251140a8ac372817a203633cf939e9a08ce3ba1', 'source_key': 'model.language_model.layers.21.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '25f36d8e983fd642e0ef9a7601809be478657b3e2aafa4769a40f6c0cb131cb8'},
    'model.language_model.layers.22.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'f8c6e12af4caa41f2c37d42379c21b05ff363b2b1f68b3bdc9497cb0a000242b', 'source_key': 'model.language_model.layers.22.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'ff21b3e9716ed9703026c486b96ec72170d8f786aaab30e48e64d2cdfda7258e'},
    'model.language_model.layers.22.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '2498efa35c121fbd8f841dd78648a28e8c01716bbe76844b87159cf3e9756f73', 'source_key': 'model.language_model.layers.22.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': 'b2b0897472d77bc8c932380ef0191c159186c226e31e07af1b8b1875edeb63a0'},
    'model.language_model.layers.22.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '91164203c660d971a3b887ba5e510933aaaffa168b27aed49ad59cf85cfe963f', 'source_key': 'model.language_model.layers.22.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '91164203c660d971a3b887ba5e510933aaaffa168b27aed49ad59cf85cfe963f'},
    'model.language_model.layers.22.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '6bb1149368785c6df3bf8a119cf82b629f4949dc88f8fc0c6e353012b3e5b259', 'source_key': 'model.language_model.layers.22.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '6bb1149368785c6df3bf8a119cf82b629f4949dc88f8fc0c6e353012b3e5b259'},
    'model.language_model.layers.22.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'a6a86feadbdd8374387712fc5fadc1f7eb4ac69f44014db628872805d72b6ee0', 'source_key': 'model.language_model.layers.22.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'a6a86feadbdd8374387712fc5fadc1f7eb4ac69f44014db628872805d72b6ee0'},
    'model.language_model.layers.22.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'b1eba21c10bcc55967d9770163db4946ed1940c96cbd347497ab6c17b12ff502', 'source_key': 'model.language_model.layers.22.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'b1eba21c10bcc55967d9770163db4946ed1940c96cbd347497ab6c17b12ff502'},
    'model.language_model.layers.22.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '27920528dfa52987e8962ebf999757e0b78d6833c6c37464079665a29842a362', 'source_key': 'model.language_model.layers.22.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '62759f952f772d3094f3fd007534d82da7025290f42a2d1850dc6d7a1a25ff19'},
    'model.language_model.layers.22.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'd51c604c4ac7d84d6c9b5a986af485727c93cca124833694214b12fe44503013', 'source_key': 'model.language_model.layers.22.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '3e41a3cc59b90ed3995fee42bbc2fef93d25867c0daacf9b7837ba8fb4a6bd37'},
    'model.language_model.layers.23.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a34d7ad484902c439d94d7933a2c61f04601c590cbf941727a86eb81b8106f0b', 'source_key': 'model.language_model.layers.23.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '658d5e3fce8f27b082447733b9a078947aeda9061b754abd03f6eb641bf9b48c'},
    'model.language_model.layers.23.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '3e82b7a89ce63389b33e9d6674d3875c58b8a1239407317bd92b1673b8c864b1', 'source_key': 'model.language_model.layers.23.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '4cf9c5d62ab86b012d034a26cf4ab169203c27ac948b7a1eaddda64ab5a4bacb'},
    'model.language_model.layers.23.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': '025e7237b2617f87458896645571e9b3076a95827dc1ff98e6df36f046637cdc', 'source_key': 'model.language_model.layers.23.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '6de7b2caa88aea2a4a880a86f869116112d7efa9dff321a9f912d1dbb037025b'},
    'model.language_model.layers.23.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'f396a3b1e600176d09fa90d34599e1721232eb3ae665f2f060a197665a151cef', 'source_key': 'model.language_model.layers.23.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '370041207173c1442311ec9fc24890da0c2f7e2ff1abf3174ce2accf6eed1d94'},
    'model.language_model.layers.24.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '4b805d4991661ee1e85646f5970778ade7361b0d06b89a998da4094dbac882b0', 'source_key': 'model.language_model.layers.24.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '603b56566986d2a4a9e415b22f26bf9cd9043ae31013ea3f9f805e2c5160b33e'},
    'model.language_model.layers.24.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '38b05fd2ba39cc1bfe0bd2b850f44d3bb3d72558fcffb86c3060df0abcd811de', 'source_key': 'model.language_model.layers.24.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': 'c0a7487dd797d8c0e0ae72d53aa5258b3fca5e0a21f220cbb5b90458d9cd58ac'},
    'model.language_model.layers.24.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'e763e26db0c39d15a26c5440ec811ddf499496758985a28fe8cb729c81e2c64a', 'source_key': 'model.language_model.layers.24.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'e763e26db0c39d15a26c5440ec811ddf499496758985a28fe8cb729c81e2c64a'},
    'model.language_model.layers.24.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '1892302b89caf70983230107dba46d725547aa2b3beebcfca0a31b1059392d74', 'source_key': 'model.language_model.layers.24.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '1892302b89caf70983230107dba46d725547aa2b3beebcfca0a31b1059392d74'},
    'model.language_model.layers.24.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'b07395f2b2094724b190e085a43d4182e5b7b937800b593f2cad9ed525e050a0', 'source_key': 'model.language_model.layers.24.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'b07395f2b2094724b190e085a43d4182e5b7b937800b593f2cad9ed525e050a0'},
    'model.language_model.layers.24.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'd5a19f9bd32263817a08a00b03a56ee4b71eba82da7b2da497233a1f27c2509f', 'source_key': 'model.language_model.layers.24.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'd5a19f9bd32263817a08a00b03a56ee4b71eba82da7b2da497233a1f27c2509f'},
    'model.language_model.layers.24.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '9be04206b45176d34ae431bf8018f4afcbdbae9fe17afd99a70957b4dade3cc9', 'source_key': 'model.language_model.layers.24.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '145182d0a351ba187ccdf1f5d018f4f234f76a907e98bf36af24c66a96bdf6bf'},
    'model.language_model.layers.24.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '4bc7acf8aa44bcb72f8fc50b9f3fd1d394aca5f845a39f91c18a38734ff11c1d', 'source_key': 'model.language_model.layers.24.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'a2823c323299c8013bcc57a84c4d94d27d58c4ab13137af91f31f37bde6fa204'},
    'model.language_model.layers.25.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '9d29fda7195abc666a5aa85cfa8980b5c8130334564ed43deb6b8343df61fd61', 'source_key': 'model.language_model.layers.25.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'c90eddf5acb4f92da60b3174716ad0eb536b38aa0c5676c9e3a7b2ab2bd6038b'},
    'model.language_model.layers.25.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'ef02c89c116a4f084643d4e8dbfb7ac17b1a0f7e9048882bc2a12afba758b59e', 'source_key': 'model.language_model.layers.25.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '63799cbca8a7c0dc4657968c67971395467f69ab390ba6813898ff8698e23d82'},
    'model.language_model.layers.25.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'e35a8326450e3364d93153b09798808b80dcada99efe28ceb569255151d209ae', 'source_key': 'model.language_model.layers.25.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'e35a8326450e3364d93153b09798808b80dcada99efe28ceb569255151d209ae'},
    'model.language_model.layers.25.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '0ba581eae9318859832564ca463b9caaf2b475c0b9a0cff8c29a46ff2d703509', 'source_key': 'model.language_model.layers.25.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '0ba581eae9318859832564ca463b9caaf2b475c0b9a0cff8c29a46ff2d703509'},
    'model.language_model.layers.25.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '8994591bbf5a6136c713fd17f42e7b3155d7f86178be5cc237e354d295555f0e', 'source_key': 'model.language_model.layers.25.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '8994591bbf5a6136c713fd17f42e7b3155d7f86178be5cc237e354d295555f0e'},
    'model.language_model.layers.25.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'a3cdc382acfed978c06c11e35c98ce20173c6c92a42537056752d46581d97e7b', 'source_key': 'model.language_model.layers.25.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'a3cdc382acfed978c06c11e35c98ce20173c6c92a42537056752d46581d97e7b'},
    'model.language_model.layers.25.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '3710bb94815e31f666f73b72618c9941de61782c82f0543b8b48e3d4ea25dc58', 'source_key': 'model.language_model.layers.25.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'edea811abb3b812e834f015eb4d45fccde1f75141ba000b847fcd4ba17d321f3'},
    'model.language_model.layers.25.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '042f07fd82ac3c07e2b92de10ed53201233f62c856a6fee6d9a9725e04ba8932', 'source_key': 'model.language_model.layers.25.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '008ced37d552f72022a633fca8e5227e822613d03e020306f2d0dd1cee9a238a'},
    'model.language_model.layers.26.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'bcc0e96bfc47d408a6bbabaefbde509cb8f4f5b98663f45750999de33bc67f64', 'source_key': 'model.language_model.layers.26.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'ddf53bd7f8f496c5208dd64908ee9c184803d7c60b9d9e0e72136c8ffd07ad4b'},
    'model.language_model.layers.26.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '24616bc58aa9f7a87ad9680a5860c925cf31d2a62d841db41b85a9b2bf0fe090', 'source_key': 'model.language_model.layers.26.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '4541f658a923a1124243c6370d95324d5f1e4bd091ce9fc2551838a96a4c980c'},
    'model.language_model.layers.26.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '0a8cad20aebb2b3ab48e6a7dcb527d54eaf4d10edf2e3e34f0066b7a5b384ccd', 'source_key': 'model.language_model.layers.26.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '0a8cad20aebb2b3ab48e6a7dcb527d54eaf4d10edf2e3e34f0066b7a5b384ccd'},
    'model.language_model.layers.26.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '95e525e9c534e849f0cf3e309b77b6ca60ec4688c73f40db7b686f10ead5bce0', 'source_key': 'model.language_model.layers.26.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '95e525e9c534e849f0cf3e309b77b6ca60ec4688c73f40db7b686f10ead5bce0'},
    'model.language_model.layers.26.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'a77b57b72f40201c1f693da54c6bd9b8626f496be613976dd001ca557b041202', 'source_key': 'model.language_model.layers.26.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'a77b57b72f40201c1f693da54c6bd9b8626f496be613976dd001ca557b041202'},
    'model.language_model.layers.26.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '762f62365824ed5b23d27ca25f49c57c31b86f6d4a878089b5840c72457d2791', 'source_key': 'model.language_model.layers.26.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '762f62365824ed5b23d27ca25f49c57c31b86f6d4a878089b5840c72457d2791'},
    'model.language_model.layers.26.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '566e4d375f2098df49f75735a612eaf5d88f08d25e1a56e5fcc13f6cfc25db66', 'source_key': 'model.language_model.layers.26.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '2c93b17fe94333a9eb7a3eea2ff0c97e4ffd922e5cd0ac35835bfe1075497b7f'},
    'model.language_model.layers.26.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a51adeaa43dc046918800486f2131589202ad80101cb00a0c8a32ffe508b7e8d', 'source_key': 'model.language_model.layers.26.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '17eef5583a7b1649789fd85c46e823512cbf0059a8fcaeb2f58c733b042b1d48'},
    'model.language_model.layers.27.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'eb54dfdb4aad967c62596778e6539ec0ef55b5a73d113c25de028ea0aa657251', 'source_key': 'model.language_model.layers.27.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'e17ce668c71f1b1d83e1f64259f0fb3413d6ee66b6741d5c291d4118acdf3857'},
    'model.language_model.layers.27.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '9a5d8c3abaf081a153962dadc3a0c32500a2880a5f8a78618bf2a826c6328f06', 'source_key': 'model.language_model.layers.27.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'a215a56332866b660f7711d9f7fa022cfc6edcb36f4c41de482b8323afe153c3'},
    'model.language_model.layers.27.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': '73e15da1f0b0dc51c25d0d35d6ccc8fe26db2111365796c79244b603df6a0b74', 'source_key': 'model.language_model.layers.27.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '605f215df234ebd6d2b8104a58490057424433c9774baafce7f50b1c7ed124fd'},
    'model.language_model.layers.27.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'c9d14e26c9a35ef288b8cac3a21d4e35722d7bb4d710c8f8e068ed0e8196a713', 'source_key': 'model.language_model.layers.27.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': 'b4a500ca1f91b8a3f20792973444707dbc315049dff6d9991fa8b66fa4174c04'},
    'model.language_model.layers.28.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'cc093fc578d1e24ceda5c799468c77112875da1b564cacfdf1134ddb8a7deeca', 'source_key': 'model.language_model.layers.28.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '7f7881e573c73ae5648e98a735edb222376426506fdd25421f2a0b96c4397dae'},
    'model.language_model.layers.28.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '665eb38b9ca29d7e35bd6f88a2072c18d7e41504ba4802e594b960f88d7a32d6', 'source_key': 'model.language_model.layers.28.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '3a57b5fd67d57b6212bb70c1b64f462ded378ba00b2cf7925f10d231c03f57e8'},
    'model.language_model.layers.28.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '4300473bb686ba72ce28cf65372544980c7777b15baec60f23a8eacce34fed15', 'source_key': 'model.language_model.layers.28.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '4300473bb686ba72ce28cf65372544980c7777b15baec60f23a8eacce34fed15'},
    'model.language_model.layers.28.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'c2b39ed03c34958b5168cd09f6ba2cdac17625101c6a44adc28318032bd85961', 'source_key': 'model.language_model.layers.28.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'c2b39ed03c34958b5168cd09f6ba2cdac17625101c6a44adc28318032bd85961'},
    'model.language_model.layers.28.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '0d9f3bfe546bda0fe5dca5b6b411834c72366148f190b79afa8e9fcd76fa76a4', 'source_key': 'model.language_model.layers.28.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '0d9f3bfe546bda0fe5dca5b6b411834c72366148f190b79afa8e9fcd76fa76a4'},
    'model.language_model.layers.28.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'ba86d960621364f3acedd343edaa448a9d86521f04916e99c3a1e454146d56cb', 'source_key': 'model.language_model.layers.28.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'ba86d960621364f3acedd343edaa448a9d86521f04916e99c3a1e454146d56cb'},
    'model.language_model.layers.28.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '9d00e3b27842dd8e0c8bf222ea31229c0df6edc4745721c831834384638689ca', 'source_key': 'model.language_model.layers.28.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '5ba00826442b377f98301bf8f131ee303c5cb69e3f61605ce5330312db32c7e2'},
    'model.language_model.layers.28.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '9db24dadea25fbd5e491af313dc5ebf422b32a4e7a94de348bc993fc2cda7628', 'source_key': 'model.language_model.layers.28.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'b7ed28083999379364f6f402cee1936ecddb9dfa2d00a07b37f7cf4535e41b75'},
    'model.language_model.layers.29.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'ee98b10d86127d84e9f4b0b6148eb8f15dee248b874b33ee3668f4a36df5742c', 'source_key': 'model.language_model.layers.29.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '4688c2be2ae2c277c013df991ecef15e72fcf0d06d09c39b5723f1204a4d4e4a'},
    'model.language_model.layers.29.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '16c73bd4d4c6d595e9eb571b89d92580db3a4115bbb2a7ee5b9f3d79871e9bab', 'source_key': 'model.language_model.layers.29.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '9baed2ee753ec456e94bd2f5c7eb2a23328b77652acc2a0d84377cae8f87bff3'},
    'model.language_model.layers.29.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'b7bed53b8e4a673223a1e7a2a2b882461f9dffe10bccb4735c878edb3c886755', 'source_key': 'model.language_model.layers.29.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'b7bed53b8e4a673223a1e7a2a2b882461f9dffe10bccb4735c878edb3c886755'},
    'model.language_model.layers.29.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'c49be1a7f661a1df61f5fc75d4326faaa1fed7f4a258590adf8c35ba8ee7cd33', 'source_key': 'model.language_model.layers.29.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'c49be1a7f661a1df61f5fc75d4326faaa1fed7f4a258590adf8c35ba8ee7cd33'},
    'model.language_model.layers.29.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '589202c4670a5fe4b7d58f6545add1a7f670a62842508ac3d3708acb8769193d', 'source_key': 'model.language_model.layers.29.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '589202c4670a5fe4b7d58f6545add1a7f670a62842508ac3d3708acb8769193d'},
    'model.language_model.layers.29.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'afd564b74b5dcc0a13e51ebe251b737ec31e98c9abc60207782db63e785b5392', 'source_key': 'model.language_model.layers.29.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'afd564b74b5dcc0a13e51ebe251b737ec31e98c9abc60207782db63e785b5392'},
    'model.language_model.layers.29.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '1b24394d2d83396148e0a063aec6760207c32fa003354f276a2a07d4eebef4f4', 'source_key': 'model.language_model.layers.29.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'e72b5f0f518b5a09c80ba465913a4eb165fdafa9190870ac19814d05ee516f6a'},
    'model.language_model.layers.29.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '1522a45354b53f7f430164bee08defcc62be426905635839bd5956d75ddbb609', 'source_key': 'model.language_model.layers.29.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'de0bfc45134493d0d819f23b88396c059c640d9d5fee178c36bfc37294ef8407'},
    'model.language_model.layers.3.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '484390b22e3a51e4efd565f6bcf99695811087a8002a865fdbebed9f8593f85c', 'source_key': 'model.language_model.layers.3.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '1a69d103970208dbf0c9cb2d21cc12771d7481a40bab05ea0a74e319bc98ea01'},
    'model.language_model.layers.3.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'f1caf6e08a7b33b2e7e13936e67d1dc6381ccb228968c9f764375db5c8d3a59a', 'source_key': 'model.language_model.layers.3.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '7ad0b8789fd0bcf4abada2645bacdf309e8b0ef7d0789a2ff80a53df9d654359'},
    'model.language_model.layers.3.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'd059576c610dc012f4a3dd81b7e11e138ace9866202d1a66495608fbf03e4514', 'source_key': 'model.language_model.layers.3.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': 'ec77a0e658adbc8c64833cdebf9be9e13f50fa7375ad61217530c54e97c5d135'},
    'model.language_model.layers.3.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'f011380d5828433da8f95630a9e64e94a32098ae5f62d6e1b8795cc6929ec88f', 'source_key': 'model.language_model.layers.3.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '6699f543299caa89ffbd9bd87d4ad8a73411bd7c77a9c43a4b712e317e51821b'},
    'model.language_model.layers.30.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a25e52c569651d50633f7a1ca806946020c8b982f7ea82b2f5d0e4271a5f4240', 'source_key': 'model.language_model.layers.30.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'cc9d49bd4781384c97117fa3ae01f71d3fdb42414ab7ee15ea2a9fb583ba4c5f'},
    'model.language_model.layers.30.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '091cd8314e40174ff6fcb1281ac122f0b06832bfe7fb4fa9a92f0b56169177b8', 'source_key': 'model.language_model.layers.30.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': 'aff0a6649d2d1250b48e7d0971ae24501072e668ea8e233183156378056fda40'},
    'model.language_model.layers.30.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '61b6f258ee917b48492d705f841f78a0dfa1ff679b3b9dee01cafd2b6595d0a2', 'source_key': 'model.language_model.layers.30.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '61b6f258ee917b48492d705f841f78a0dfa1ff679b3b9dee01cafd2b6595d0a2'},
    'model.language_model.layers.30.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '731a2ae1adc2810a4ae473624620e34ec9f096473ab1926f61bf4e4b4a50197a', 'source_key': 'model.language_model.layers.30.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '731a2ae1adc2810a4ae473624620e34ec9f096473ab1926f61bf4e4b4a50197a'},
    'model.language_model.layers.30.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '5d6d2d129fe5aa85f1ca77706afcad76fb0604208a675c98d46d1c9169854605', 'source_key': 'model.language_model.layers.30.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '5d6d2d129fe5aa85f1ca77706afcad76fb0604208a675c98d46d1c9169854605'},
    'model.language_model.layers.30.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'ee757ac8d76df3b475b3d4788c54c9451a501b689a914eafc76260cd7b1ec716', 'source_key': 'model.language_model.layers.30.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'ee757ac8d76df3b475b3d4788c54c9451a501b689a914eafc76260cd7b1ec716'},
    'model.language_model.layers.30.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '5a0e68ea6c8b12a8b2b47edd27f2b551503fec1906b1e1f821538f6e87a4a99a', 'source_key': 'model.language_model.layers.30.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '148e0b309cec3c60f1bec801bd8d61f097f11e15520978be333e7547d2e8e040'},
    'model.language_model.layers.30.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '2a791a4043b91d86f81436befcc69137aa0d999db2bcc55fb9a6fa5bdaeacd87', 'source_key': 'model.language_model.layers.30.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'daeee4a7f9226dfd698fdea61152b36e76a178f4537d87edb870e4d28600bb93'},
    'model.language_model.layers.31.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'ea794d3d841ff6d262b20a02dc9e2b93ce5a82b2ec5ff46ee7d27038a9a7cd6b', 'source_key': 'model.language_model.layers.31.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '256f3b3a5c7ec72959392746e82c6baf03b3ba1e96f4becfd918c7a445bf8866'},
    'model.language_model.layers.31.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '7339c70d3e91b9e4d431415a05e961ddbeb3d596376f7a29cb799201b0c9b1c7', 'source_key': 'model.language_model.layers.31.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '7202d29a8a2fc7f7bc8da8fe838339ef241d9009140940eaf8d4ca4cf94d9c8a'},
    'model.language_model.layers.31.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'a8e6241efbdf00591fc6d05ee41ab3c620690fd2a3e0fb9ff928eece0b6d2b51', 'source_key': 'model.language_model.layers.31.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '53215d4bcfef7be8b1c39c8ab7416f83492db400f80e95b9a025daaef16440d1'},
    'model.language_model.layers.31.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'd6ac09e7ec0aef3a3be6a94c7d65c80e43d145c0330d67780663bf1a867ee964', 'source_key': 'model.language_model.layers.31.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': '8ab2a095a52bfe3aacc76df750342e48ebe522a81cac27e5670f30477052a950'},
    'model.language_model.layers.4.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'e984790dc120e215068bc55c18956a308ce2856d840c1739f50ab03eb7b1b19c', 'source_key': 'model.language_model.layers.4.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '920866c20286e9edcd7bf58e356bede3b002f6ad44dbf451d8d96c6d3cdf1c7b'},
    'model.language_model.layers.4.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '4d850cda96852cb20aad43aa6b0734719888c80abf4157c181fa06105bb242ff', 'source_key': 'model.language_model.layers.4.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '72ea956f2465d2ef7684cf67d83b2c519ae261dd720a650d06b57701e7437b4c'},
    'model.language_model.layers.4.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'c7e4f593c6ca655007deb8a8066d7312849ebc353e0bd224cbaef5e6387a0e02', 'source_key': 'model.language_model.layers.4.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'c7e4f593c6ca655007deb8a8066d7312849ebc353e0bd224cbaef5e6387a0e02'},
    'model.language_model.layers.4.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'b32d6b03c8fe63abd22bc33ce4153f57ddaf81d0f78c9d9d94f3f3c6d5d55d01', 'source_key': 'model.language_model.layers.4.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'b32d6b03c8fe63abd22bc33ce4153f57ddaf81d0f78c9d9d94f3f3c6d5d55d01'},
    'model.language_model.layers.4.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '4b4f994cc5f4db5fa510ac68c3fabbdf08b7e3c6d27ca573d31eeccaee6c0661', 'source_key': 'model.language_model.layers.4.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '4b4f994cc5f4db5fa510ac68c3fabbdf08b7e3c6d27ca573d31eeccaee6c0661'},
    'model.language_model.layers.4.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'b9c8e6c3e626a814af0b8522c6f485d9359268f8d2db41a9a7f1ce89741df9a2', 'source_key': 'model.language_model.layers.4.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'b9c8e6c3e626a814af0b8522c6f485d9359268f8d2db41a9a7f1ce89741df9a2'},
    'model.language_model.layers.4.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '9b57778a436975b2ab3d9c4981cf8617ba5c2f69fbbc9eefab6a0dbc10951bf4', 'source_key': 'model.language_model.layers.4.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '08052478dc082ee28f810cba407b1b31e60fa05982bd8ff23e23277f5977882f'},
    'model.language_model.layers.4.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '53730d79f869c3a2f4a4328371c522f420bfd5b87617a336e94a9a2e25d3a803', 'source_key': 'model.language_model.layers.4.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'fdbb91230859aebd6fbadf75dd319f09e5e4688e765519eedece93baed2c256f'},
    'model.language_model.layers.5.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a27c344fb6044374e0ed572b876a8b9d3899d65a585cc98458cf88488b29f9f1', 'source_key': 'model.language_model.layers.5.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'fdd00320aae329d2cddf7d62cb02e96bf9fed7784a807812e0353e8849d10d0d'},
    'model.language_model.layers.5.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '416e85f2e527200f1464a7f364c728cb758b19ddb2213b1513b8555ff76427a6', 'source_key': 'model.language_model.layers.5.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '0628dae0aa108d403b97e04fb3f6f2d40c8b5c64605e62d30c96487116b2e2ce'},
    'model.language_model.layers.5.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'd2da1d32ae544a0cdf4931f450ca5c7c300f1eb98531c257d52635dfde3a918c', 'source_key': 'model.language_model.layers.5.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'd2da1d32ae544a0cdf4931f450ca5c7c300f1eb98531c257d52635dfde3a918c'},
    'model.language_model.layers.5.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '46d4ba2cb65c6c6079bd970a6fa6dfd4e076f44cface4708e4a4224df4f11031', 'source_key': 'model.language_model.layers.5.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '46d4ba2cb65c6c6079bd970a6fa6dfd4e076f44cface4708e4a4224df4f11031'},
    'model.language_model.layers.5.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '0a9da2090a7da81b5ed35843c72aa5b4100a38fc2ca987c91405c26f7af765ed', 'source_key': 'model.language_model.layers.5.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '0a9da2090a7da81b5ed35843c72aa5b4100a38fc2ca987c91405c26f7af765ed'},
    'model.language_model.layers.5.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'f489d0875402098c11542d63cff76ddfd664a0b3453073071e7ff3f43d943822', 'source_key': 'model.language_model.layers.5.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'f489d0875402098c11542d63cff76ddfd664a0b3453073071e7ff3f43d943822'},
    'model.language_model.layers.5.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '8b980512429c9b5e6d813ca9270938dc6b2625476f4147670b30d8d5aa764356', 'source_key': 'model.language_model.layers.5.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '40fac5ec8926f9a34c8178d686ca5a33403f9703b2ce01badcc2e31d46c64dac'},
    'model.language_model.layers.5.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'f3297a5c441e5f7b0865e4148d349b26af9149b709aeba91a8666c96c1d47eb9', 'source_key': 'model.language_model.layers.5.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'c910bfef198cf31ccbd0af19b6be855694656daefd010fc16eebcfa3147b7726'},
    'model.language_model.layers.6.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'cefae69a01c8cffd8507872eab589415385d82d0ba1df6836510343754255a33', 'source_key': 'model.language_model.layers.6.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '1ed35557df04ca3a1e54eea3f6a74aeb1bd7fa628e82ce7cb2e96d7ab7df163c'},
    'model.language_model.layers.6.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': '3cc7d0bb01bfd1b4bc78ded22f31ff28fcbafc5d27d45a7454add9de583da1bf', 'source_key': 'model.language_model.layers.6.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '6f1a0300d572af80fa919e2b119056eb36db0d9ece014bee367aa319526ca8dd'},
    'model.language_model.layers.6.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': '50c833940f650c0d89eca6d3d20ef01e8cd2531abba56900d265051a5f290290', 'source_key': 'model.language_model.layers.6.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': '50c833940f650c0d89eca6d3d20ef01e8cd2531abba56900d265051a5f290290'},
    'model.language_model.layers.6.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '89440090bb614134939338ac03bd372c778e6f65e3fc440ab23f80b7ccafd735', 'source_key': 'model.language_model.layers.6.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '89440090bb614134939338ac03bd372c778e6f65e3fc440ab23f80b7ccafd735'},
    'model.language_model.layers.6.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '15f439db92bba4953bb79f72c04160c561dbf43fa47961a82a2f0fec5ab7edde', 'source_key': 'model.language_model.layers.6.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '15f439db92bba4953bb79f72c04160c561dbf43fa47961a82a2f0fec5ab7edde'},
    'model.language_model.layers.6.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '7648e22d3e0ab0db3620c3516effefec4ede9ef68499e542554b65ca9a3a7a66', 'source_key': 'model.language_model.layers.6.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '7648e22d3e0ab0db3620c3516effefec4ede9ef68499e542554b65ca9a3a7a66'},
    'model.language_model.layers.6.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '49acefccd5e289c32282c5152121caf42a55f0d89e75f3615f7c8b1589eb1d6b', 'source_key': 'model.language_model.layers.6.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': 'cab3b34193c2058f95180f77a18520ad4dbb10b9dce1e46a33af9bde09da74b1'},
    'model.language_model.layers.6.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '05f78498a1dad530be8b09735718fb72012ab7d4d6702c1d519a6cc518afc3b5', 'source_key': 'model.language_model.layers.6.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '2807f5bf064d37b06d264102a9a275ffa38343679a9f6bb4b4b78bac98f66a07'},
    'model.language_model.layers.7.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '9c6faa84fc3c21d87c0ec18102b650fa689a554e3b74f169185f2995826ddfa6', 'source_key': 'model.language_model.layers.7.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '3866accc357167c6e5446b661f4e62db165f0760ed50e1a9d6f0d61a99ed19b8'},
    'model.language_model.layers.7.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'ba80e07aa380810b4bc4641d50cc68221c4e2f5725780137829abf42354b621d', 'source_key': 'model.language_model.layers.7.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'aabecb51965d3f0f73be5ec5916ff79ea37e133cfb13461edbb053f3179175fa'},
    'model.language_model.layers.7.self_attn.k_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': 'e8b531904687ddaedac30abb059f386a19049d1750948d8e5178011e6e1930c2', 'source_key': 'model.language_model.layers.7.self_attn.k_norm.weight', 'source_dtype': 'BF16', 'source_sha256': 'b1dd41369843ec99312e35aa5f890720509d690293d919f859595cbe105f08d3'},
    'model.language_model.layers.7.self_attn.q_norm.weight': {'shape': [256], 'dtype': 'BF16', 'sha256': '23a935c5e7d3e2a365886f23f24382b185360225446fc320ea2ecbc3dda69922', 'source_key': 'model.language_model.layers.7.self_attn.q_norm.weight', 'source_dtype': 'BF16', 'source_sha256': 'efa89bb96f0f662d07b6d960f8a2ee462712248dce1cbb7be7ce4faa4c8b3389'},
    'model.language_model.layers.8.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'edb0b839e88317a8b89dad6b6dd552845f017045e75ef9421553c92b3e2b5bcd', 'source_key': 'model.language_model.layers.8.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'd8aaf6cf90402b527d344d8ecd632bea13a280bf3ddc8dfd19f9fa23e53cdf2a'},
    'model.language_model.layers.8.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'e318e579afc11d58aa5c2d7f0986adfa63dcac210a6c01024a4cba9edd1e8370', 'source_key': 'model.language_model.layers.8.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '720472e02dfd59a126e668df9696d2e81507c52a00ca5b2975d143fbba0612ff'},
    'model.language_model.layers.8.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'f4a2425971914a9e330e01d7729a3a059dbaf97cd14d2317c22909c6ad6f9978', 'source_key': 'model.language_model.layers.8.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'f4a2425971914a9e330e01d7729a3a059dbaf97cd14d2317c22909c6ad6f9978'},
    'model.language_model.layers.8.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': 'eed9b488aa45bf7d628f5b842edadeb4c67f88622e5f5c444cd3d561f0d3255b', 'source_key': 'model.language_model.layers.8.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': 'eed9b488aa45bf7d628f5b842edadeb4c67f88622e5f5c444cd3d561f0d3255b'},
    'model.language_model.layers.8.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'c96a080104bc608e3b004bc342aca3b260d637b33b3ce59de44266e9135caed8', 'source_key': 'model.language_model.layers.8.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': 'c96a080104bc608e3b004bc342aca3b260d637b33b3ce59de44266e9135caed8'},
    'model.language_model.layers.8.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': 'a0a3138ac00da07a8c9a37ce4f048467c6f02f3624ab9e4ce07a75ff45b2526a', 'source_key': 'model.language_model.layers.8.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': 'a0a3138ac00da07a8c9a37ce4f048467c6f02f3624ab9e4ce07a75ff45b2526a'},
    'model.language_model.layers.8.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '45f6ebbc8d5e2ca992af14a0f060990dc5fa63d1ac942ac10951802fc5b0d066', 'source_key': 'model.language_model.layers.8.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '8ab56b9c98eedb84f26019d6f4ddb55082ec26070dd5e31fe92081f566dab969'},
    'model.language_model.layers.8.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '1b1431716511b5f0abd36780ee848b8cf81c5d9237eafe20f4fb3426876bceeb', 'source_key': 'model.language_model.layers.8.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '567af8d0e477dc6064dabf608177b9e9600bddd783542599a840b4f922c3763b'},
    'model.language_model.layers.9.input_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'c5beec67e561e73427ff52d463831025b9ae7856ab8d95648d5752b450e9e7c1', 'source_key': 'model.language_model.layers.9.input_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': 'da44fff59881241e99c0217ca60634c021ae8865e10bcf10b59097c7a332fcf1'},
    'model.language_model.layers.9.linear_attn.A_log': {'shape': [32], 'dtype': 'BF16', 'sha256': 'e7137b42a9fe36b2b5a88abd0c48e474077aee7d521034a78c2d74c7ca66f560', 'source_key': 'model.language_model.layers.9.linear_attn.A_log', 'source_dtype': 'F32', 'source_sha256': '33786924d8c98f8c14f9ceb2da9dbc25bd8122b9def1475bf16c655c7c2e258b'},
    'model.language_model.layers.9.linear_attn.conv1d.weight': {'shape': [8192, 1, 4], 'dtype': 'BF16', 'sha256': 'f8820249da364817dd26af398351ffb5ff84f6529a14163ce6f884045871c835', 'source_key': 'model.language_model.layers.9.linear_attn.conv1d.weight', 'source_dtype': 'BF16', 'source_sha256': 'f8820249da364817dd26af398351ffb5ff84f6529a14163ce6f884045871c835'},
    'model.language_model.layers.9.linear_attn.dt_bias': {'shape': [32], 'dtype': 'BF16', 'sha256': '62ba12b1fa4fbfca5e40cbc30aac7788a1d74555896f1bbca7b68c504817505e', 'source_key': 'model.language_model.layers.9.linear_attn.dt_bias', 'source_dtype': 'BF16', 'source_sha256': '62ba12b1fa4fbfca5e40cbc30aac7788a1d74555896f1bbca7b68c504817505e'},
    'model.language_model.layers.9.linear_attn.in_proj_a.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '05546c3beac837672491145d0099e2b34515b2cc1597be7e91b7fda3d9fe7600', 'source_key': 'model.language_model.layers.9.linear_attn.in_proj_a.weight', 'source_dtype': 'BF16', 'source_sha256': '05546c3beac837672491145d0099e2b34515b2cc1597be7e91b7fda3d9fe7600'},
    'model.language_model.layers.9.linear_attn.in_proj_b.weight': {'shape': [32, 2560], 'dtype': 'BF16', 'sha256': '15c6e085c3c2d366313f243b8887533680a8571f7df94a870d3bfe2358f21f7e', 'source_key': 'model.language_model.layers.9.linear_attn.in_proj_b.weight', 'source_dtype': 'BF16', 'source_sha256': '15c6e085c3c2d366313f243b8887533680a8571f7df94a870d3bfe2358f21f7e'},
    'model.language_model.layers.9.linear_attn.norm.weight': {'shape': [128], 'dtype': 'BF16', 'sha256': '6d2a7dbd3ff19ac8939a129448d6fdd235d7f784d3c6cb6654d6f8df58c9b153', 'source_key': 'model.language_model.layers.9.linear_attn.norm.weight', 'source_dtype': 'F32', 'source_sha256': '5746c004068e91e86e35986859bff7800a153ba72cb4aa62ecb8aa1ba2a1a4b2'},
    'model.language_model.layers.9.post_attention_layernorm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': '8091df3a3bcf31042cfb83a193745ea326cee3777eb47d8d1fee78563bd606ea', 'source_key': 'model.language_model.layers.9.post_attention_layernorm.weight', 'source_dtype': 'BF16', 'source_sha256': '1d52fea89f4925b72d5c3dfc3e8687f1c59eb2ed7d4ab64c4cf3029332176e09'},
    'model.language_model.norm.weight': {'shape': [2560], 'dtype': 'BF16', 'sha256': 'a5892083080d8a7b7be5a0f5343e497346bd1698e9c2ead52943f5f612c2a468', 'source_key': 'model.language_model.norm.weight', 'source_dtype': 'BF16', 'source_sha256': '96294c653d4e8ab158cdae0b7e794559654ffdc62db880725ed714d47f708621'},
}

L0_REPLACEMENTS = {
    'model.language_model.layers.0.mlp.gate_proj.weight': '6cf68b4f363bbaa42f33bcf975e9ed8a6cde737d645833652c3b1db019973992',
    'model.language_model.layers.0.mlp.up_proj.weight': 'f7b4258c00e6114aed743741fdc685df7a64077cd59308de3c7ae0d560de4e2f',
}

HF_ASSET_HASHES = {
    'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
    'generation_config.json': '62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73',
    'tokenizer.json': '06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
    'tokenizer_config.json': '66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b',
}

KV_SCALES = [
    {'layer_index': 3, 'module_name': 'model.layers.3.self_attn.attn', 'k_scale': 0.03130580484867096, 'v_scale': 0.02792968787252903},
    {'layer_index': 7, 'module_name': 'model.layers.7.self_attn.attn', 'k_scale': 0.04296875, 'v_scale': 0.04266183078289032},
    {'layer_index': 11, 'module_name': 'model.layers.11.self_attn.attn', 'k_scale': 0.03959263488650322, 'v_scale': 0.06199776753783226},
    {'layer_index': 15, 'module_name': 'model.layers.15.self_attn.attn', 'k_scale': 0.03959263488650322, 'v_scale': 0.0598493292927742},
    {'layer_index': 19, 'module_name': 'model.layers.19.self_attn.attn', 'k_scale': 0.04787946492433548, 'v_scale': 0.1227678582072258},
    {'layer_index': 23, 'module_name': 'model.layers.23.self_attn.attn', 'k_scale': 0.03207310289144516, 'v_scale': 0.07857143133878708},
    {'layer_index': 27, 'module_name': 'model.layers.27.self_attn.attn', 'k_scale': 0.04419642686843872, 'v_scale': 0.09882812201976776},
    {'layer_index': 31, 'module_name': 'model.layers.31.self_attn.attn', 'k_scale': 0.03099888376891613, 'v_scale': 0.1043526753783226},
]


_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8,
                "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                "F8_E4M3": 1, "F8_E4M3FN": 1, "F8_E5M2": 1}


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)


def _regions(path, begin, length):
    with path.open("rb") as stream:
        stream.seek(begin)
        while length:
            chunk = stream.read(min(length, 4 * 2**20))
            if not chunk:
                raise ValueError(f"Truncated tensor payload: {path.name}")
            yield chunk
            length -= len(chunk)


def file_sha256(path):
    digest = hashlib.sha256()
    for chunk in _regions(Path(path), 0, Path(path).stat().st_size):
        digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(item):
    begin, end = item["data_offsets"]
    digest = hashlib.sha256()
    for chunk in _regions(item["path"], item["base"] + begin, end - begin):
        digest.update(chunk)
    return digest.hexdigest()


def scan_safetensors(path):
    """Validate header, dtype/shape sizes and dense non-overlapping payloads."""
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Short safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 64 * 2**20:
            raise ValueError("Invalid safetensors header length")
        header = stream.read(length)
        if len(header) != length:
            raise ValueError("Short safetensors header")
    header = json.loads(header, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    base, size = 8 + length, path.stat().st_size
    result = {}
    spans = []
    for key, item in header.items():
        if key == "__metadata__":
            if not isinstance(item, dict) or not all(isinstance(v, str) for v in item.values()):
                raise ValueError("Safetensors metadata must contain string values")
            continue
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor header fields: {key}")
        shape, offsets, dtype = item["shape"], item["data_offsets"], item["dtype"]
        if (not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets) or dtype not in _DTYPE_BYTES):
            raise ValueError(f"Invalid tensor dtype/shape/offsets: {key}")
        begin, end = offsets
        if not 0 <= begin <= end <= size - base or end - begin != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise ValueError(f"Tensor payload size differs from dtype/shape: {key}")
        spans.append((begin, end))
        result[key] = {"path": path, "base": base, **item}
    position = 0
    for begin, end in sorted(spans):
        if begin != position:
            raise ValueError("Safetensors payload has gaps or overlapping tensors")
        position = end
    if position != size - base:
        raise ValueError("Safetensors payload has trailing bytes")
    return result


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError("Checkpoint index paths must be relative and contained")
    return path


def scan_checkpoint(root):
    root = Path(root)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("Checkpoint has no safetensors shards")
    tensors = {}
    for path in shards:
        for key, item in scan_safetensors(path).items():
            if key in tensors:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            tensors[key] = item
    indices = sorted(root.glob("*.safetensors.index.json"))
    if len(indices) > 1 or (len(shards) > 1 and not indices):
        raise ValueError("Sharded checkpoint requires one safetensors index")
    index = read_json(indices[0]) if indices else None
    if index is not None:
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or set(mapping) != set(tensors):
            raise ValueError("Checkpoint index does not exactly cover its tensors")
        for key, filename in mapping.items():
            if root / _safe_relative(filename) != tensors[key]["path"]:
                raise ValueError(f"Checkpoint index points to the wrong shard: {key}")
        total = sum(item["data_offsets"][1] - item["data_offsets"][0] for item in tensors.values())
        if index.get("metadata", {}).get("total_size") != total:
            raise ValueError("Checkpoint index total_size differs from payload bytes")
    return read_json(root / "config.json"), tensors, indices[0] if indices else None


def resolve_bf16_key(key, tensors):
    """Map canonical text keys to explicitly supported BF16 release prefixes."""
    if key.startswith("model.language_model."):
        suffix = key.removeprefix("model.language_model.")
        candidates = (key, "model.language_model.model." + suffix,
                      "language_model.model." + suffix, "model." + suffix)
    else:
        candidates = (key,)
    matches = [candidate for candidate in candidates if candidate in tensors]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous BF16 source key: {key}")
    return matches[0]


def selected_keys():
    return {f"model.language_model.layers.{layer}.self_attn.{proj}.weight"
            for layer in LAYERS for proj in ("q_proj", "k_proj", "v_proj", "o_proj")}


def scale_key(key):
    return key.removesuffix(".weight") + ".weight_scale"


def _require_identity(key, item, shape, dtype, digest):
    valid_dtype = item["dtype"] in ("F8_E4M3", "F8_E4M3FN") if dtype == "E4M3" else item["dtype"] == dtype
    if item["shape"] != shape or not valid_dtype or tensor_sha256(item) != digest:
        raise ValueError(f"Fixed tensor shape/dtype/hash differs: {key}")


def _check_geometry(config):
    text = config.get("text_config", config)
    required = {"model_type": "qwen3_5_text", "num_hidden_layers": 32, "hidden_size": 2560,
                "intermediate_size": 9216, "vocab_size": 248320, "tie_word_embeddings": True,
                "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "full_attention_interval": 4, "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32", "rms_norm_eps": 1e-6,
                "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True,
                "hidden_act": "silu", "max_position_embeddings": 262144, "eos_token_id": 248044}
    if any(text.get(key) != value for key, value in required.items()):
        raise ValueError("Fixed Qwen3.5-4B configuration geometry differs")
    layers = ["full_attention" if i in LAYERS else "linear_attention" for i in range(32)]
    if text.get("layer_types") != layers:
        raise ValueError("Fixed model attention-layer order differs")
    rope = text.get("rope_parameters", {})
    if any(rope.get(key) != value for key, value in {
            "mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "default",
            "rope_theta": 10000000, "partial_rotary_factor": 0.25}.items()):
        raise ValueError("Fixed model RoPE configuration differs")


_MXFP8_QUANTIZATION_RULE = {
    "actorder": None, "block_structure": None, "group_size": 32, "num_bits": 8,
    "observer_kwargs": {}, "scale_dtype": "torch.uint8", "strategy": "group",
    "symmetric": True, "type": "float", "zp_dtype": None,
}
_MXFP8_QUANTIZATION_CONFIG = {
    "config_groups": {"group_0": {
        "format": "mxfp8-quantized", "targets": ["Linear"], "output_activations": None,
        "weights": {**_MXFP8_QUANTIZATION_RULE, "dynamic": False, "observer": "memoryless_minmax"},
        "input_activations": {**_MXFP8_QUANTIZATION_RULE, "dynamic": True, "observer": None},
    }},
    "format": "mxfp8-quantized", "global_compression_ratio": None,
    "kv_cache_scheme": None, "quant_method": "compressed-tensors",
    "quantization_status": "compressed", "sparsity_config": {},
    "transform_config": {}, "version": "0.17.0",
}


def _prepare_config(config):
    new = copy.deepcopy(config)
    sections = [new]
    if isinstance(new.get("text_config"), dict):
        sections.append(new["text_config"])
    found = False
    expected_ignore = {f"model.layers.{i}.linear_attn.in_proj_{proj}"
                       for i in range(32) if i not in LAYERS for proj in ("a", "b")} | {"lm_head"}
    additions = sorted({f"model.layers.{i}.self_attn.{proj}" for i in LAYERS for proj in ("qkv_proj", "o_proj")})
    for section in sections:
        quant = section.get("quantization_config")
        if quant is None:
            continue
        found = True
        if quant.get("quant_method") != "compressed-tensors" or quant.get("format") != "mxfp8-quantized":
            raise ValueError("Expected compressed-tensors MXFP8 parent configuration")
        ignores = quant.get("ignore")
        if not isinstance(ignores, list) or len(ignores) != len(expected_ignore) or set(ignores) != expected_ignore:
            raise ValueError("MXFP8 parent ignore set is not the fixed base model")
        # Bind the complete accepted schema, including target selection,
        # dynamic/symmetric semantics and optional transforms. Canonical JSON
        # also distinguishes bools from integers; ignore ordering is checked
        # separately because adding the BF16 attention boundary is this edit.
        scheme = {name: value for name, value in quant.items() if name != "ignore"}
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical(scheme) != canonical(_MXFP8_QUANTIZATION_CONFIG):
            raise ValueError("MXFP8 quantization configuration differs from the fixed profile")
        quant["ignore"] = ignores + additions
    if not found:
        raise ValueError("MXFP8 quantization configuration is absent")
    return new


def _source_tree(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Checkpoint input must be a real directory")
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Checkpoint input contains a symlink")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError("Checkpoint input contains a non-regular file")
    return sorted(files)


def _copy_shard(source, target, replacements, removed):
    items = scan_safetensors(source)
    header = {}
    with source.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        metadata = json.loads(stream.read(length), object_pairs_hook=_unique_object).get("__metadata__")
    if metadata is not None:
        header["__metadata__"] = metadata
    position = 0
    for key, prior in items.items():
        if key in removed:
            continue
        item = replacements.get(key, prior)
        length = item["data_offsets"][1] - item["data_offsets"][0]
        header[key] = {"dtype": item["dtype"], "shape": item["shape"],
                       "data_offsets": [position, position + length]}
        position += length
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    with target.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for key, prior in items.items():
            if key not in removed:
                item = replacements.get(key, prior)
                begin, end = item["data_offsets"]
                for chunk in _regions(item["path"], item["base"] + begin, end - begin):
                    stream.write(chunk)


def _write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _publish(stage, output):
    """Atomically rename with no-replace semantics, including empty directories."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux" and hasattr(libc, "renameat2"):
        result = libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(os.fsencode(stage), os.fsencode(output), 4)
    else:
        raise RuntimeError("Atomic no-replace publication requires Linux renameat2 or macOS renamex_np")
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))


def prepare_model(bf16, mxfp8=None, l0_codes=None, output=None):
    """Create a new fixed-profile checkpoint; never overwrite an output path.

    Inputs are verified against embedded tensor identities. The returned
    manifest records current file and tensor hashes without absolute paths.
    Source acquisition and redistribution permission remain unverified here.
    """
    if l0_codes is None or output is None:
        raise ValueError("l0_codes and output are required")
    if mxfp8 is None:
        from .reconstruct_model import reconstruct_model
        return reconstruct_model(bf16, l0_codes, output)
    paths = [Path(p).expanduser().absolute() for p in (bf16, mxfp8, l0_codes, output)]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Checkpoint inputs and output must not be symlinks")
    bf16, mxfp8, l0_codes, output = (path.resolve() for path in paths)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must exist")
    for source in (bf16, mxfp8):
        if source == output or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be independent of both checkpoint inputs")
    source_files = _source_tree(bf16)
    parent_files = _source_tree(mxfp8)
    if l0_codes.is_dir():
        l0_codes = l0_codes / "replacement_weights.safetensors"
    if l0_codes.is_symlink() or not l0_codes.is_file():
        raise ValueError("L0 codes must be a real safetensors file")
    if l0_codes.stat().st_size != L0_PAYLOAD_BYTES or file_sha256(l0_codes) != L0_PAYLOAD_SHA256:
        raise ValueError("Fixed L0 safetensors payload size/hash differs")
    # The verified export used safetensors.torch.save_file, not torch.save.
    # Arbitrary pickle/state_dict loading is intentionally unsupported.
    replacement = scan_safetensors(l0_codes)
    if set(replacement) != set(L0_REPLACEMENTS):
        raise ValueError("L0 codes must contain exactly gate_proj and up_proj")
    source_config, source, _ = scan_checkpoint(bf16)
    parent_config, parent, index_path = scan_checkpoint(mxfp8)
    _check_geometry(source_config)
    _check_geometry(parent_config)
    new_config = _prepare_config(parent_config)
    expected_parent = set(RETAINED) | set(QUANTIZED) | {scale_key(key) for key in QUANTIZED}
    if set(parent) != expected_parent:
        raise ValueError("MXFP8 parent tensor inventory differs from the fixed champion base")
    source_keys = set()
    for key, row in QUANTIZED.items():
        mapped = resolve_bf16_key(key, source)
        source_keys.add(mapped)
        _require_identity(mapped, source[mapped], row["shape"], "BF16", row["source_sha256"])
        _require_identity(key, parent[key], row["shape"], "E4M3", row["stored_values_sha256"])
        n, k = row["shape"]
        _require_identity(scale_key(key), parent[scale_key(key)], [n, k // 32], "U8", row["stored_scales_sha256"])
    for key, row in RETAINED.items():
        mapped = resolve_bf16_key(row["source_key"], source)
        source_keys.add(mapped)
        _require_identity(mapped, source[mapped], row["shape"], row["source_dtype"], row["source_sha256"])
        _require_identity(key, parent[key], row["shape"], row["dtype"], row["sha256"])
    for key, digest in L0_REPLACEMENTS.items():
        _require_identity(key, replacement[key], QUANTIZED[key]["shape"], "E4M3", digest)
    for name, digest in HF_ASSET_HASHES.items():
        if not (mxfp8 / name).is_file() or file_sha256(mxfp8 / name) != digest:
            raise ValueError(f"Fixed tokenizer/template/generation artifact differs: {name}")
    if not source_keys <= set(source):
        raise ValueError("BF16 source key mapping is incomplete")
    selected = selected_keys()
    if not selected <= set(QUANTIZED):
        raise ValueError("Fixed full-attention identity inventory is incomplete")
    removed = {scale_key(key) for key in selected}
    replacements = {key: source[resolve_bf16_key(key, source)] for key in selected}
    replacements.update(replacement)
    expected_total = sum((replacements.get(key, item)["data_offsets"][1] - replacements.get(key, item)["data_offsets"][0])
                         for key, item in parent.items() if key not in removed)
    copy_bytes = sum(path.stat().st_size for path in parent_files) + sum(
        (replacements[key]["data_offsets"][1] - replacements[key]["data_offsets"][0])
        - (parent[key]["data_offsets"][1] - parent[key]["data_offsets"][0]) for key in selected)
    if shutil.disk_usage(output.parent).free < copy_bytes:
        raise OSError("Insufficient disk space for an independent checkpoint copy")
    # Hash the actual inputs; no stored audit file or source path is a proof.
    inputs = {
        "bf16": {"provenance": "source_pending_verification", "files": [
            {"name": str(path.relative_to(bf16)), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in source_files if path.suffix == ".safetensors" or path.name == "config.json"]},
        "mxfp8": {"provenance": "source_pending_verification", "files": [
            {"name": str(path.relative_to(mxfp8)), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in parent_files if path.suffix == ".safetensors" or path.name in HF_ASSET_HASHES or path.name == "config.json"]},
        "l0_codes": {"provenance": "source_pending_verification", "format": "safetensors",
                     "sha256": L0_PAYLOAD_SHA256, "bytes": L0_PAYLOAD_BYTES},
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    published = False
    try:
        copied = []
        keep = set(HF_ASSET_HASHES) | {"config.json"}
        keep.update(str(item["path"].relative_to(mxfp8)) for item in parent.values())
        if index_path is not None:
            keep.add(str(index_path.relative_to(mxfp8)))
        for path in parent_files:
            relative = str(path.relative_to(mxfp8))
            if relative not in keep:
                continue
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == "config.json":
                _write_json(target, new_config)
                kind = "full_attention_ignore_added"
            elif path.suffix == ".safetensors":
                here = {key: value for key, value in replacements.items() if parent[key]["path"] == path}
                removed_here = {key for key in removed if parent[key]["path"] == path}
                if here or removed_here:
                    _copy_shard(path, target, here, removed_here)
                    kind = "restored_bf16_and_l0_codes"
                else:
                    shutil.copy2(path, target)
                    kind = "unchanged"
            elif path == index_path:
                index = read_json(path)
                index["weight_map"] = {key: value for key, value in index["weight_map"].items() if key not in removed}
                index["metadata"]["total_size"] = expected_total
                _write_json(target, index)
                kind = "updated_tensor_index"
            else:
                shutil.copy2(path, target)
                kind = "unchanged"
            before, after = file_sha256(path), file_sha256(target)
            if kind == "unchanged" and before != after:
                raise ValueError(f"Copied asset bytes changed: {relative}")
            copied.append({"name": relative, "kind": kind, "input_sha256": before,
                           "sha256": after, "bytes": target.stat().st_size})
        candidate_config, candidate, _ = scan_checkpoint(stage)
        if candidate_config != new_config or set(candidate) != set(parent) - removed:
            raise ValueError("Output config or tensor inventory differs from the plan")
        tensors = []
        for key, item in candidate.items():
            expected = replacements.get(key, parent[key])
            digest = tensor_sha256(item)
            if item["shape"] != expected["shape"] or item["dtype"] != expected["dtype"] or digest != tensor_sha256(expected):
                raise ValueError(f"Output tensor byte verification failed: {key}")
            kind = "restored_bf16" if key in selected else "asymmetric_l0_code" if key in L0_REPLACEMENTS else "unchanged"
            tensors.append({"key": key, "dtype": item["dtype"], "shape": item["shape"],
                            "sha256": digest, "kind": kind})
        profile = {"profile": PROFILE, "kv_scales": copy.deepcopy(KV_SCALES)}
        _write_json(stage / "mach_profile.json", profile)
        copied.append({"name": "mach_profile.json", "kind": "fixed_profile_metadata",
                       "sha256": file_sha256(stage / "mach_profile.json"),
                       "bytes": (stage / "mach_profile.json").stat().st_size})
        manifest = {"status": "complete", "profile": PROFILE,
                    "scope": "checkpoint byte materialization; loader and GPU fidelity unqualified",
                    "source_provenance": "source_pending_verification",
                    "inputs": inputs, "output_files": copied, "output_tensors": tensors,
                    "restored_bf16_weights": sorted(selected), "removed_scale_keys": sorted(removed),
                    "l0_replacement_keys": sorted(L0_REPLACEMENTS),
                    "unmodified_tensor_bytes_verified": True, "tensor_payload_bytes": expected_total,
                    "materializer_sha256": file_sha256(Path(__file__))}
        _write_json(stage / MANIFEST_NAME, manifest)
        validate_model(stage, verify_weights=False)
        _publish(stage, output)
        published = True
        return manifest
    finally:
        if not published:
            shutil.rmtree(stage)


def _expected_output_tensors():
    expected = {key: {"dtype": row["dtype"], "shape": row["shape"], "sha256": row["sha256"]}
                for key, row in RETAINED.items()}
    restored = selected_keys()
    for key, row in QUANTIZED.items():
        expected[key] = {"dtype": "BF16" if key in restored else "E4M3", "shape": row["shape"],
                         "sha256": row["source_sha256"] if key in restored else
                         L0_REPLACEMENTS.get(key, row["stored_values_sha256"])}
        if key not in restored:
            n, k = row["shape"]
            expected[scale_key(key)] = {"dtype": "U8", "shape": [n, k // 32],
                                        "sha256": row["stored_scales_sha256"]}
    return expected


def validate_model(model_dir, *, verify_weights=False):
    """Validate a materialized fixed profile and return its metadata.

    Header/config/profile/tokenizer/manifest binding is always checked. Set
    verify_weights=True for full file and tensor byte hashing at cold startup;
    False does not certify model payload integrity. Neither verifies provenance.
    """
    model_dir = Path(model_dir).expanduser().absolute()
    _source_tree(model_dir)
    config, tensors, _ = scan_checkpoint(model_dir)
    _check_geometry(config)
    original = copy.deepcopy(config)
    additions = {f"model.layers.{i}.self_attn.{proj}" for i in LAYERS for proj in ("qkv_proj", "o_proj")}
    sections = [original]
    if isinstance(original.get("text_config"), dict):
        sections.append(original["text_config"])
    for section in sections:
        if "quantization_config" in section:
            ignores = section["quantization_config"].get("ignore", [])
            if not additions <= set(ignores):
                raise ValueError("Materialized config lacks BF16 full-attention ignores")
            section["quantization_config"]["ignore"] = [name for name in ignores if name not in additions]
    if _prepare_config(original) != config:
        raise ValueError("Materialized quantization configuration differs")
    profile = read_json(model_dir / "mach_profile.json")
    if profile != {"profile": PROFILE, "kv_scales": KV_SCALES}:
        raise ValueError("Materialized profile or frozen FP32 KV scales differ")
    manifest = read_json(model_dir / MANIFEST_NAME)
    if (manifest.get("status") != "complete" or manifest.get("profile") != PROFILE
            or manifest.get("restored_bf16_weights") != sorted(selected_keys())
            or manifest.get("removed_scale_keys") != sorted(scale_key(key) for key in selected_keys())
            or manifest.get("l0_replacement_keys") != sorted(L0_REPLACEMENTS)):
        raise ValueError("Materialization manifest contract differs")
    rows = manifest.get("output_tensors")
    if not isinstance(rows, list) or len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Materialization tensor manifest is invalid or duplicated")
    manifest_tensors = {row["key"]: row for row in rows}
    expected = _expected_output_tensors()
    if set(tensors) != set(expected) or set(manifest_tensors) != set(expected):
        raise ValueError("Materialized tensor inventory differs from the fixed profile")
    for key, identity in expected.items():
        item, row = tensors[key], manifest_tensors[key]
        valid_dtype = item["dtype"] in ("F8_E4M3", "F8_E4M3FN") if identity["dtype"] == "E4M3" else item["dtype"] == identity["dtype"]
        if (not valid_dtype or item["shape"] != identity["shape"]
                or row["shape"] != item["shape"] or row["dtype"] != item["dtype"]
                or row["sha256"] != identity["sha256"]):
            raise ValueError(f"Materialized tensor header/manifest identity differs: {key}")
        if verify_weights and tensor_sha256(item) != identity["sha256"]:
            raise ValueError(f"Materialized tensor bytes differ: {key}")
    files = manifest.get("output_files")
    if not isinstance(files, list) or len({row["name"] for row in files}) != len(files):
        raise ValueError("Materialization file manifest is invalid or duplicated")
    by_name = {row["name"]: row for row in files}
    actual = {str(path.relative_to(model_dir)) for path in _source_tree(model_dir)}
    if actual != set(by_name) | {MANIFEST_NAME}:
        raise ValueError("Materialized file set differs from its manifest")
    for name, row in by_name.items():
        path = model_dir / _safe_relative(name)
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"Materialized file size differs: {name}")
        if verify_weights or path.suffix != ".safetensors":
            if file_sha256(path) != row["sha256"]:
                raise ValueError(f"Materialized file hash differs: {name}")
    for name, digest in HF_ASSET_HASHES.items():
        if name not in by_name or file_sha256(model_dir / name) != digest:
            raise ValueError(f"Fixed tokenizer/template/generation artifact differs: {name}")
    total = sum(item["data_offsets"][1] - item["data_offsets"][0] for item in tensors.values())
    if total != manifest.get("tensor_payload_bytes"):
        raise ValueError("Materialized tensor payload total differs")
    metadata = copy.deepcopy(profile)
    metadata["materialization_manifest_sha256"] = file_sha256(model_dir / MANIFEST_NAME)
    metadata["verified_weights"] = bool(verify_weights)
    metadata["source_provenance"] = "source_pending_verification"
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--mxfp8", type=Path, help="Optional verified original MXFP8 checkpoint; otherwise reconstruct from BF16")
    parser.add_argument("--l0-codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_model(args.bf16, args.mxfp8, args.l0_codes, args.output)
    print(json.dumps({"status": result["status"], "profile": result["profile"],
                      "tensor_count": len(result["output_tensors"])}))


if __name__ == "__main__":
    main()
