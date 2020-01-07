This is a short description how to run the demo on the `neuralmonkey-czm`
virtual machine.

First, run models as separate services in the `monkey-env` environment. The
easiest and the dirtiest way is to run them separate screen windows.

```bash
neuralmonkey-sentiment/bin/neuralmonkey-server --configuration demo_models/yelp_rnn_san/run.ini --port 4000
neuralmonkey-sentiment/bin/neuralmonkey-server --configuration demo_models/csfd_rnn_san/run.ini --port 4001
neuralmonkey/bin/neuralmonkey-server --configuration demo_models/encs-transormer/experiment.ini --preprocess demo_models/encs-transormer/preprocess.ini --port 4002
neuralmonkey/bin/neuralmonkey-server --configuration demo_models/resnet/run.ini --port 4003
neuralmonkey-captioning/bin/neuralmonkey-server --configuration demo_models/captioning_cs_bigger/run.ini --port 4004
neuralmonkey-captioning/bin/neuralmonkey-server --configuration demo_models/captioning_en_multiref_bigger/run.ini --port 4005
```

The web app can be then run with (port 3000 is mapped outsied):

```bash
./server.py --port 3000 --host 0.0.0.0 --sentiment-en localhost:4000 --sentiment-cs localhost:4001 --translation-encs localhost:4002 --resnet localhost:03 --captioning-en localhost:4004 --captioning-cs localhost:4005
```

The models used in this demo are permanently store at
[Lindat](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2839).

The model are run from different code bases of Neural Monkey. In particular
they are on commits:

| repository                   | commit
| :--------------------------- | :------------
| `neuralmonkey`               | `f4409f0778882c5d36288` 
| `neuralmonkey-sentinement`   | `d019f261e0725567a7742` 
