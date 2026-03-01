
// ============================================================================
// TESTING THE INTEGRATION
// ============================================================================

# Test 1: Check device connectivity
curl -X POST http://192.168.1.100:5000/device/connect

# Test 2: Send command to device
curl -X POST http://192.168.1.100:5000/device/command \
  -H "Content-Type: application/json" \
  -d '{"command": "F"}'

# Test 3: Get camera snapshot
curl http://192.168.1.100:5000/camera/snapshot -o snapshot.jpg

# Test 4: Start cleaning
curl -X POST http://192.168.1.100:5000/cleaning/start \
  -H "Content-Type: application/json" \
  -d '{"duration": 120}'

# Test 5: Check system status
curl http://192.168.1.100:5000/status

# Test 6: Switch to autonomous mode
curl -X POST http://192.168.1.100:5000/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "AUTONOMOUS"}'

# Test 7: Run full pipeline
curl -X POST http://192.168.1.100:5000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{}'

