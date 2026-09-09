curl -X POST "http://pc:8000/normal/generate" ^
     -H "X-API-Key: 1" ^
     -F "file=@test.png" ^
     --output test_result.png