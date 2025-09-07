# API Scaling Guide: Wait Time Prediction Service

This guide shows how to scale the wait time prediction service horizontally for high-throughput microservices architecture.

## 🏗️ Architecture Overview

The API service is designed to handle the combinatorial explosion in microservices by:
- **Batch Processing**: Process multiple predictions in single requests
- **Parallel Execution**: Use ThreadPoolExecutor for CPU-bound ML tasks
- **Async Support**: Background jobs for large batch processing
- **Optimization Endpoints**: Reduce N×M service calls to single optimized assignment

## 🚀 Quick Start

### 1. Start Single Instance
```bash
# Install dependencies
pip install -r api/requirements.txt

# Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test with Client Examples
```bash
python api/client_examples.py
```

## 📈 Scaling Strategies

### Horizontal Scaling with Docker

#### Single Instance
```bash
docker build -f api/Dockerfile -t wait-time-api .
docker run -p 8000:8000 wait-time-api
```

#### Multiple Instances with Load Balancer
```bash
# Scale to 3 instances
docker-compose -f docker-compose.api.yml up --scale wait-time-api=3

# Access through nginx load balancer on port 80
curl http://localhost/health
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wait-time-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: wait-time-api
  template:
    metadata:
      labels:
        app: wait-time-api
    spec:
      containers:
      - name: api
        image: wait-time-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "4Gi" 
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: wait-time-api-service
spec:
  selector:
    app: wait-time-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔥 Performance Optimization

### 1. Batch Processing Benefits

**Before: Individual Service Calls**
```
Routes Service → Wait Time Service (Call 1)
Routes Service → Wait Time Service (Call 2)  
Routes Service → Wait Time Service (Call 3)
... (N calls for N routes)
```

**After: Single Batch Call**
```
Routes Service → Wait Time Service (Batch of N predictions)
```

**Performance Gains:**
- Reduce network overhead by ~90%
- Batch tensor operations for better CPU utilization
- Single model load per batch vs N model loads

### 2. Driver Optimization Benefits

**Before: Combinatorial Service Calls**
```
For 10 jobs × 5 drivers = 50 individual prediction calls
Selection Service makes 50 API calls
Then applies optimization logic
```

**After: Single Optimization Call**
```
Single API call with jobs + available drivers
Returns optimal assignments directly
```

**Performance Gains:**
- Reduce 50 API calls to 1
- Server-side optimization with full context
- Built-in batch processing for all combinations

### 3. Concurrent Processing

The API uses:
- **ThreadPoolExecutor**: For CPU-bound ML inference
- **AsyncIO**: For I/O-bound operations
- **Batch Tensor Operations**: Process multiple predictions simultaneously

## 📊 API Endpoints

### Core Prediction Endpoints

| Endpoint | Purpose | Batch Size | Use Case |
|----------|---------|------------|-----------|
| `/predict/single` | Single prediction | 1 | Real-time user requests |
| `/predict/batch` | Batch predictions | 1-10,000 | Route planning, bulk analysis |
| `/predict/async` | Async batch job | 1-10,000 | Large-scale processing |

### Optimization Endpoints

| Endpoint | Purpose | Complexity Reduction |
|----------|---------|---------------------|
| `/optimize/drivers` | Driver assignment | O(N×M) → O(1) API calls |
| `/predict/bulk-scenarios` | Scenario analysis | Hundreds of calls → 1 call |

### Monitoring Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Service health check |
| `/model/info` | Model metadata |
| `/jobs/{job_id}` | Async job status |

## 🎯 Usage Patterns

### Pattern 1: Route Planning Service
```python
# Replace multiple individual calls
routes = get_possible_routes()
predictions = []
for route in routes:
    pred = call_wait_time_service(route)  # ❌ N API calls
    predictions.append(pred)

# With single batch call
routes = get_possible_routes()
batch_request = [convert_route_to_request(r) for r in routes]
predictions = api_client.predict_batch(batch_request)  # ✅ 1 API call
```

### Pattern 2: Driver Assignment Service
```python
# Replace combinatorial explosion
best_assignments = {}
for job in jobs:
    for driver in available_drivers:
        prediction = call_wait_time_service(job, driver)  # ❌ N×M calls
        # ... optimization logic
    best_assignments[job] = best_driver

# With optimization endpoint
result = api_client.optimize_drivers(jobs, available_drivers)  # ✅ 1 API call
assignments = result['job_assignments']
```

### Pattern 3: Analytics Dashboard
```python
# Replace many scenario calls
scenario_results = []
for scenario in scenarios:
    for sample in generate_samples(scenario):
        prediction = call_wait_time_service(sample)  # ❌ Hundreds of calls
    scenario_results.append(aggregate_results(predictions))

# With bulk scenario endpoint  
result = api_client.bulk_scenario_analysis(scenarios)  # ✅ 1 API call
scenario_results = result['results']
```

## 🏋️ Load Testing

### Test Batch Performance
```python
import time
import concurrent.futures

def benchmark_batch_vs_individual(n_requests=100):
    client = WaitTimePredictionClient()
    
    # Generate test requests
    requests = generate_test_requests(n_requests)
    
    # Method 1: Individual calls
    start_time = time.time()
    individual_results = []
    for req in requests:
        result = client.predict_single(**req)
        individual_results.append(result)
    individual_time = time.time() - start_time
    
    # Method 2: Batch call
    start_time = time.time()
    batch_result = client.predict_batch(requests)
    batch_time = time.time() - start_time
    
    print(f"Individual calls: {individual_time:.2f}s")
    print(f"Batch call: {batch_time:.2f}s") 
    print(f"Speedup: {individual_time/batch_time:.1f}x")
```

### Concurrent Load Test
```bash
# Using Apache Bench
ab -n 1000 -c 10 -T application/json -p batch_request.json http://localhost:8000/predict/batch

# Using wrk
wrk -t4 -c100 -d30s --script=batch_test.lua http://localhost:8000/predict/batch
```

## 📈 Monitoring & Metrics

### Key Metrics to Track
- **Throughput**: Predictions per second
- **Latency**: P50, P95, P99 response times
- **Batch Size Distribution**: Optimal batch sizes
- **Error Rate**: Failed predictions
- **Resource Usage**: CPU, Memory per instance

### Health Check Integration
```python
def service_health_check():
    """Integrate with your service discovery."""
    try:
        response = requests.get("http://wait-time-api/health", timeout=5)
        return response.json().get("model_loaded", False)
    except:
        return False
```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
WORKERS=4                    # Number of worker processes
MAX_BATCH_SIZE=10000        # Maximum batch size
MODEL_PATH=/app/models/     # Model file location
REDIS_URL=redis://redis:6379  # Cache connection

# Resource Limits
MAX_MEMORY_MB=4096          # Memory limit per instance
CPU_LIMIT=2.0               # CPU limit per instance
```

### Auto-scaling Configuration (Kubernetes)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wait-time-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wait-time-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🚀 Deployment Checklist

- [ ] Model files available in container/volume
- [ ] Health checks configured
- [ ] Load balancer routing rules set
- [ ] Auto-scaling policies defined
- [ ] Monitoring dashboards created
- [ ] Circuit breakers implemented
- [ ] Rate limiting configured
- [ ] Logging aggregation setup

## 💡 Cost Optimization Tips

1. **Right-size instances**: Start with 2 CPU, 4GB RAM per instance
2. **Use batch processing**: 10-100 predictions per batch for optimal throughput
3. **Cache frequent requests**: Implement Redis caching for common scenarios
4. **Auto-scale down**: Scale to minimum instances during low traffic
5. **Spot instances**: Use spot instances for non-critical workloads

---

**Result**: This architecture reduces microservice complexity from O(N×M) individual API calls to O(1) optimized batch operations, dramatically improving performance and reducing costs.