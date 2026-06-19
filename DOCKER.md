# Docker запуск Django-сервиса

Сборка и запуск:

```powershell
docker compose up --build
```

Открыть сайт:

```text
http://127.0.0.1:8000/
```

Остановка:

```powershell
docker compose down
```

Пересоздать базу, если нужно:

```powershell
docker compose down
Remove-Item service/db.sqlite3
docker compose up --build
```

Контейнер использует CPU-compatible режим. Веса моделей подключаются из `./outputs`, а Django-сервис, media-файлы и SQLite база подключаются из `./service`.
