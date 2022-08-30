# PPLM Chatbot
Репозриторий содержит в себе реализацию чатбота с подходом [Plug and Play Language Models](https://github.com/uber-research/PPLM) (Uber Research)

Чатбот основывается на архитектуре GPT2. Основная задача заключалась в оптимизации и адаптации подхода PPLM для выбранной архитектуры

Файл **run_pplm.py** содержит в себе скрипт для интеракции с ботом

Файл **src/pplm.py** содержит в себе улучшенную версию [алгоритма](https://github.com/uber-research/PPLM/blob/master/run_pplm.py) из оригинального репозитория авторов

Во время реализации проекта решались следующие задачи:
- реализация архитектуры чатбота
- реализация кода обучения
- реализация и оптимизация алгоритма PPLM
- обучение нескольких моделей аттрибутов для PPLM
- обучение самого чатбота