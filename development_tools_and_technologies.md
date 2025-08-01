# Development Tools and Technologies – Docker, GitHub, GitHub Actions, pytest, APIs, GraphDB

## Docker

### What is Docker?

**Definition**: Docker is a platform for developing, shipping, and running applications in containers.

**Core Concept**: Containerization allows applications to run consistently across different environments by packaging the application and its dependencies together.

### Docker Architecture

#### **Key Components**:

**Docker Engine**:
- **Daemon**: Background service managing containers
- **CLI**: Command-line interface for Docker commands
- **REST API**: API for Docker operations

**Docker Images**:
- **Definition**: Read-only templates for creating containers
- **Layers**: Multiple layers that can be shared between images
- **Registry**: Storage and distribution of images (Docker Hub, private registries)

**Docker Containers**:
- **Definition**: Running instances of Docker images
- **Isolation**: Each container runs in its own isolated environment
- **Portability**: Can run on any system with Docker installed

### Dockerfile Best Practices

#### **Basic Dockerfile Structure**:
```dockerfile
# Use official base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "app.py"]
```

#### **Optimization Techniques**:
- **Multi-stage Builds**: Reduce final image size
- **Layer Caching**: Optimize build times
- **Minimal Base Images**: Use slim or alpine images
- **Security**: Run as non-root user

#### **Advanced Dockerfile Example**:
```dockerfile
# Multi-stage build for Python application
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "app.py"]
```

### Docker Compose

#### **Basic docker-compose.yml**:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db
    volumes:
      - .:/app

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

#### **Advanced Features**:
- **Service Dependencies**: Control startup order
- **Environment Variables**: Configuration management
- **Volume Mounting**: Persistent data storage
- **Network Configuration**: Service communication

### Docker Best Practices

#### **Security**:
- **Non-root Users**: Run containers as non-root
- **Image Scanning**: Scan for vulnerabilities
- **Secrets Management**: Use Docker secrets or external tools
- **Minimal Images**: Reduce attack surface

#### **Performance**:
- **Layer Optimization**: Minimize layers and size
- **Build Caching**: Leverage Docker layer caching
- **Resource Limits**: Set memory and CPU limits
- **Health Checks**: Monitor container health

#### **Production Considerations**:
- **Orchestration**: Use Kubernetes or Docker Swarm
- **Monitoring**: Implement logging and metrics
- **Backup**: Regular image and data backups
- **CI/CD Integration**: Automated builds and deployments

## GitHub

### What is GitHub?

**Definition**: GitHub is a web-based platform for version control and collaboration using Git.

**Core Features**: Code hosting, issue tracking, pull requests, project management, and CI/CD integration.

### Git Fundamentals

#### **Basic Git Commands**:
```bash
# Initialize repository
git init

# Clone repository
git clone <repository-url>

# Add files to staging
git add <filename>
git add .

# Commit changes
git commit -m "Commit message"

# Push to remote
git push origin main

# Pull latest changes
git pull origin main

# Check status
git status

# View commit history
git log
```

#### **Branching and Merging**:
```bash
# Create new branch
git checkout -b feature-branch

# Switch branches
git checkout main

# Merge branch
git merge feature-branch

# Delete branch
git branch -d feature-branch
```

### GitHub Workflow

#### **Collaborative Development**:
1. **Fork Repository**: Create personal copy
2. **Clone Fork**: Download to local machine
3. **Create Branch**: Work on feature in separate branch
4. **Make Changes**: Implement features or fixes
5. **Commit Changes**: Save work with descriptive messages
6. **Push to Fork**: Upload changes to personal fork
7. **Create Pull Request**: Request merge to original repository
8. **Code Review**: Team reviews and provides feedback
9. **Merge**: Integrate changes into main branch

#### **Pull Request Best Practices**:
- **Descriptive Titles**: Clear, concise titles
- **Detailed Descriptions**: Explain what and why
- **Small Changes**: Keep PRs focused and manageable
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs when necessary

### GitHub Features

#### **Issues and Projects**:
- **Issue Tracking**: Bug reports and feature requests
- **Labels and Milestones**: Organization and categorization
- **Project Boards**: Kanban-style project management
- **Templates**: Standardized issue and PR templates

#### **Security Features**:
- **Dependabot**: Automated dependency updates
- **Code Scanning**: Security vulnerability detection
- **Secret Scanning**: Detect exposed secrets
- **Branch Protection**: Enforce code review and testing

## GitHub Actions

### What are GitHub Actions?

**Definition**: GitHub Actions is a CI/CD platform that automates software workflows directly in GitHub repositories.

**Core Concept**: Define workflows using YAML files that trigger on repository events and execute automated tasks.

### Workflow Structure

#### **Basic Workflow**:
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest
```

#### **Advanced Workflow**:
```yaml
name: Advanced CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.9'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    - run: pip install flake8 black
    - run: flake8 .
    - run: black --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - run: pip install -r requirements.txt
    - run: pytest --cov=app

  deploy:
    needs: [lint, test]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to production
      run: echo "Deploying to production"
```

### GitHub Actions Features

#### **Triggers and Events**:
- **Push/Pull Request**: Code changes
- **Schedule**: Time-based triggers
- **Manual**: Manual workflow execution
- **External**: Webhook triggers

#### **Runners and Environments**:
- **GitHub-hosted**: Ubuntu, Windows, macOS
- **Self-hosted**: Custom runners
- **Matrix Strategy**: Multiple configurations
- **Environment Secrets**: Secure configuration

#### **Advanced Features**:
- **Artifacts**: Share data between jobs
- **Caching**: Speed up builds
- **Conditional Steps**: Execute based on conditions
- **Parallel Jobs**: Concurrent execution

## pytest

### What is pytest?

**Definition**: pytest is a testing framework for Python that makes it easy to write simple and scalable test cases.

**Key Features**: Simple syntax, powerful fixtures, parameterized testing, and extensive plugin ecosystem.

### Basic pytest Usage

#### **Simple Test Example**:
```python
# test_example.py
def test_addition():
    assert 2 + 2 == 4

def test_string_concatenation():
    result = "Hello" + " " + "World"
    assert result == "Hello World"

def test_list_operations():
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
```

#### **Test Organization**:
```python
# test_calculator.py
class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5
    
    def test_subtract(self):
        calc = Calculator()
        assert calc.subtract(5, 3) == 2
    
    def test_multiply(self):
        calc = Calculator()
        assert calc.multiply(4, 3) == 12
```

### pytest Fixtures

#### **Basic Fixtures**:
```python
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_with_fixture(sample_data):
    assert len(sample_data) == 5
    assert sum(sample_data) == 15

@pytest.fixture
def database_connection():
    # Setup
    connection = create_database_connection()
    yield connection
    # Teardown
    connection.close()
```

#### **Advanced Fixtures**:
```python
import pytest

@pytest.fixture(scope="session")
def database():
    """Session-scoped database fixture"""
    db = create_test_database()
    yield db
    cleanup_test_database(db)

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Automatically run for each test"""
    # Setup
    setup_test_environment()
    yield
    # Teardown
    cleanup_test_environment()

@pytest.fixture(params=[1, 2, 3])
def number_fixture(request):
    """Parameterized fixture"""
    return request.param
```

### pytest Features

#### **Parameterized Testing**:
```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (5, 25)
])
def test_square(input, expected):
    assert input ** 2 == expected

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (5, 5, 10),
    (-1, 1, 0)
])
def test_addition(a, b, expected):
    assert a + b == expected
```

#### **Markers and Categories**:
```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    # This test is marked as slow
    pass

@pytest.mark.integration
def test_database_integration():
    # Integration test
    pass

@pytest.mark.unit
def test_unit_function():
    # Unit test
    pass
```

#### **Assertions and Debugging**:
```python
def test_complex_assertions():
    result = complex_function()
    
    # Detailed assertions
    assert result is not None
    assert isinstance(result, dict)
    assert "key" in result
    assert result["key"] == "expected_value"
    
    # Using pytest's assertion introspection
    assert len(result) == 3, f"Expected 3 items, got {len(result)}"
```

### pytest Configuration

#### **pytest.ini**:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

#### **conftest.py**:
```python
import pytest

@pytest.fixture(scope="session")
def global_fixture():
    """Available to all tests in the session"""
    return "global_value"

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
```

## APIs

### What are APIs?

**Definition**: Application Programming Interfaces (APIs) are sets of rules and protocols that allow different software applications to communicate with each other.

**Types**: REST APIs, GraphQL APIs, SOAP APIs, gRPC APIs.

### REST API Design

#### **REST Principles**:
- **Stateless**: Each request contains all necessary information
- **Client-Server**: Separation of concerns
- **Cacheable**: Responses can be cached
- **Uniform Interface**: Consistent resource identification and manipulation

#### **HTTP Methods**:
```python
# GET - Retrieve data
GET /api/users
GET /api/users/123

# POST - Create new resource
POST /api/users
{
    "name": "John Doe",
    "email": "john@example.com"
}

# PUT - Update entire resource
PUT /api/users/123
{
    "name": "John Doe",
    "email": "john@example.com"
}

# PATCH - Partial update
PATCH /api/users/123
{
    "email": "newemail@example.com"
}

# DELETE - Remove resource
DELETE /api/users/123
```

#### **Status Codes**:
- **2xx Success**: 200 OK, 201 Created, 204 No Content
- **3xx Redirection**: 301 Moved, 304 Not Modified
- **4xx Client Error**: 400 Bad Request, 401 Unauthorized, 404 Not Found
- **5xx Server Error**: 500 Internal Server Error, 503 Service Unavailable

### API Implementation with Flask

#### **Basic Flask API**:
```python
from flask import Flask, request, jsonify
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class UserResource(Resource):
    def get(self, user_id=None):
        if user_id:
            # Get specific user
            user = get_user(user_id)
            if user:
                return user, 200
            return {"error": "User not found"}, 404
        else:
            # Get all users
            users = get_all_users()
            return {"users": users}, 200
    
    def post(self):
        data = request.get_json()
        user = create_user(data)
        return user, 201
    
    def put(self, user_id):
        data = request.get_json()
        user = update_user(user_id, data)
        if user:
            return user, 200
        return {"error": "User not found"}, 404
    
    def delete(self, user_id):
        if delete_user(user_id):
            return "", 204
        return {"error": "User not found"}, 404

api.add_resource(UserResource, '/api/users', '/api/users/<int:user_id>')

if __name__ == '__main__':
    app.run(debug=True)
```

#### **Advanced API Features**:
```python
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_cors import CORS
from marshmallow import Schema, fields

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
jwt = JWTManager(app)
CORS(app)

# Schema validation
class UserSchema(Schema):
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    age = fields.Int()

user_schema = UserSchema()

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = authenticate_user(data['username'], data['password'])
    if user:
        access_token = create_access_token(identity=user.id)
        return {'access_token': access_token}, 200
    return {'error': 'Invalid credentials'}, 401

@app.route('/api/users', methods=['POST'])
@jwt_required()
def create_user():
    try:
        data = user_schema.load(request.get_json())
        user = create_user(data)
        return user_schema.dump(user), 201
    except ValidationError as e:
        return {'errors': e.messages}, 400

@app.errorhandler(404)
def not_found(error):
    return {'error': 'Resource not found'}, 404

@app.errorhandler(500)
def internal_error(error):
    return {'error': 'Internal server error'}, 500
```

### API Testing

#### **pytest with requests**:
```python
import pytest
import requests

class TestUserAPI:
    BASE_URL = "http://localhost:5000/api"
    
    def test_get_users(self):
        response = requests.get(f"{self.BASE_URL}/users")
        assert response.status_code == 200
        assert "users" in response.json()
    
    def test_create_user(self):
        user_data = {
            "name": "Test User",
            "email": "test@example.com"
        }
        response = requests.post(f"{self.BASE_URL}/users", json=user_data)
        assert response.status_code == 201
        assert response.json()["name"] == "Test User"
    
    def test_get_user_not_found(self):
        response = requests.get(f"{self.BASE_URL}/users/999")
        assert response.status_code == 404
    
    @pytest.mark.parametrize("user_data,expected_status", [
        ({"name": "Valid User", "email": "valid@example.com"}, 201),
        ({"name": "Invalid User"}, 400),  # Missing email
        ({}, 400),  # Empty data
    ])
    def test_create_user_validation(self, user_data, expected_status):
        response = requests.post(f"{self.BASE_URL}/users", json=user_data)
        assert response.status_code == expected_status
```

## GraphDB

### What is GraphDB?

**Definition**: GraphDB is a type of database that uses graph structures for semantic queries with nodes, edges, and properties to represent and store data.

**Key Features**: Relationship-centric data modeling, complex queries, and semantic reasoning capabilities.

### Graph Database Concepts

#### **Core Components**:
- **Nodes (Vertices)**: Entities in the graph
- **Edges (Relationships)**: Connections between nodes
- **Properties**: Attributes of nodes and edges
- **Labels**: Categories for nodes and relationships

#### **Graph Types**:
- **Property Graphs**: Nodes and edges with properties
- **RDF Graphs**: Subject-predicate-object triples
- **Knowledge Graphs**: Semantic relationships and reasoning

### Neo4j Implementation

#### **Basic Cypher Queries**:
```cypher
// Create nodes
CREATE (john:Person {name: 'John Doe', age: 30})
CREATE (jane:Person {name: 'Jane Smith', age: 25})
CREATE (company:Company {name: 'Tech Corp'})

// Create relationships
CREATE (john)-[:WORKS_FOR]->(company)
CREATE (jane)-[:WORKS_FOR]->(company)
CREATE (john)-[:KNOWS]->(jane)

// Query nodes
MATCH (p:Person) RETURN p

// Query relationships
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
RETURN p.name, c.name

// Complex queries
MATCH (p1:Person)-[:KNOWS]->(p2:Person)-[:WORKS_FOR]->(c:Company)
WHERE p1.name = 'John Doe'
RETURN p2.name, c.name
```

#### **Advanced Cypher Patterns**:
```cypher
// Pattern matching
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
WHERE p.age > 25
RETURN p.name, c.name

// Aggregation
MATCH (c:Company)<-[:WORKS_FOR]-(p:Person)
RETURN c.name, count(p) as employee_count

// Path finding
MATCH path = (start:Person)-[:KNOWS*1..3]->(end:Person)
WHERE start.name = 'John Doe' AND end.name = 'Jane Smith'
RETURN path

// Graph algorithms
MATCH (p:Person)
CALL gds.pageRank.stream('myGraph', {nodeLabels: ['Person']})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name as name, score
ORDER BY score DESC
```

### Python Integration with Neo4j

#### **Basic Connection**:
```python
from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_person(self, name, age):
        with self.driver.session() as session:
            result = session.run(
                "CREATE (p:Person {name: $name, age: $age}) RETURN p",
                name=name, age=age
            )
            return result.single()
    
    def get_person(self, name):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Person {name: $name}) RETURN p",
                name=name
            )
            return result.single()
    
    def create_relationship(self, person1_name, person2_name, relationship_type):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p1:Person {name: $person1_name})
                MATCH (p2:Person {name: $person2_name})
                CREATE (p1)-[r:$relationship_type]->(p2)
                RETURN r
                """,
                person1_name=person1_name,
                person2_name=person2_name,
                relationship_type=relationship_type
            )
            return result.single()

# Usage
neo4j = Neo4jConnection("bolt://localhost:7687", "neo4j", "password")
neo4j.create_person("John Doe", 30)
neo4j.create_relationship("John Doe", "Jane Smith", "KNOWS")
```

#### **Advanced Graph Operations**:
```python
class GraphOperations:
    def __init__(self, driver):
        self.driver = driver
    
    def find_shortest_path(self, start_name, end_name):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (start:Person {name: $start_name})-[*]-(end:Person {name: $end_name})
                )
                RETURN path
                """,
                start_name=start_name, end_name=end_name
            )
            return result.single()
    
    def get_recommendations(self, person_name, limit=5):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {name: $person_name})-[:KNOWS]->(friend:Person)-[:KNOWS]->(recommendation:Person)
                WHERE NOT (p)-[:KNOWS]->(recommendation)
                RETURN recommendation.name, count(friend) as mutual_friends
                ORDER BY mutual_friends DESC
                LIMIT $limit
                """,
                person_name=person_name, limit=limit
            )
            return list(result)
    
    def community_detection(self):
        with self.driver.session() as session:
            result = session.run(
                """
                CALL gds.louvain.stream('myGraph')
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).name as name, communityId
                ORDER BY communityId, name
                """
            )
            return list(result)
```

### Graph Database Use Cases

#### **Social Networks**:
- **Friend Recommendations**: Find mutual connections
- **Influence Analysis**: Identify key influencers
- **Community Detection**: Find groups and clusters

#### **Knowledge Graphs**:
- **Semantic Search**: Find related concepts
- **Reasoning**: Infer new relationships
- **Data Integration**: Connect disparate data sources

#### **Recommendation Systems**:
- **Collaborative Filtering**: Find similar users
- **Content-based**: Find similar items
- **Hybrid Approaches**: Combine multiple strategies

### Cypher Query Context Retrieval – Enhancing LLM Capabilities with Neo4j Graph DB

#### **What is Cypher Query Context Retrieval?**

**Definition**: Cypher Query Context Retrieval is a technique that uses Neo4j's Cypher query language to extract relevant context from graph databases to enhance LLM responses with real-time, structured knowledge.

**Core Concept**: By leveraging graph relationships and semantic connections, LLMs can access dynamic, contextual information that goes beyond their training data, enabling more accurate and up-to-date responses.

#### **Why Context Retrieval is Important for LLMs**:

**LLM Limitations**:
- **Static Knowledge**: Models are trained on data from a specific time period
- **Limited Context Window**: Cannot access all relevant information
- **Hallucination**: Generate information not grounded in current reality
- **Lack of Real-time Data**: Cannot access current information

**Benefits of Graph-based Context Retrieval**:
- **Dynamic Knowledge**: Access to real-time, up-to-date information
- **Semantic Relationships**: Leverage graph connections for better understanding
- **Reduced Hallucination**: Ground responses in verifiable graph data
- **Scalable Context**: Handle large knowledge graphs efficiently

#### **Cypher Query Patterns for Context Retrieval**:

**Basic Context Retrieval**:
```cypher
// Retrieve related entities for context
MATCH (entity:Entity {name: $entity_name})
MATCH (entity)-[r]-(related)
RETURN entity, type(r) as relationship, related
LIMIT 10

// Example: Get context for a person
MATCH (person:Person {name: 'John Doe'})
MATCH (person)-[r]-(related)
RETURN person.name, type(r) as relationship, 
       labels(related) as related_type, related.name
```

**Semantic Context Retrieval**:
```cypher
// Find semantically related concepts
MATCH (start:Concept {name: $concept})
MATCH path = (start)-[*1..3]-(related:Concept)
WHERE related.name <> start.name
RETURN related.name, length(path) as distance
ORDER BY distance
LIMIT 20

// Example: Find related topics for AI
MATCH (ai:Topic {name: 'Artificial Intelligence'})
MATCH path = (ai)-[*1..3]-(related:Topic)
WHERE related.name <> 'Artificial Intelligence'
RETURN related.name, length(path) as distance
ORDER BY distance
LIMIT 15
```

**Multi-hop Context Retrieval**:
```cypher
// Retrieve context through multiple relationships
MATCH (start:Entity {name: $entity_name})
MATCH path = (start)-[*1..5]-(end:Entity)
WHERE end.name <> start.name
WITH start, end, path, length(path) as distance
ORDER BY distance
RETURN end.name, distance, 
       [node in nodes(path) | node.name] as path_nodes
LIMIT 10
```

#### **Advanced Context Retrieval Techniques**:

**Weighted Context Retrieval**:
```cypher
// Retrieve context with relationship weights
MATCH (start:Entity {name: $entity_name})
MATCH (start)-[r]-(related:Entity)
WITH start, related, r.weight as weight
ORDER BY weight DESC
RETURN related.name, weight, type(r) as relationship
LIMIT 20

// Example with custom scoring
MATCH (person:Person {name: 'John Doe'})
MATCH (person)-[r]-(related)
WITH person, related, r,
     CASE type(r)
       WHEN 'WORKS_FOR' THEN 10
       WHEN 'KNOWS' THEN 5
       WHEN 'LIVES_IN' THEN 3
       ELSE 1
     END as importance
ORDER BY importance DESC
RETURN related.name, type(r) as relationship, importance
```

**Temporal Context Retrieval**:
```cypher
// Retrieve context based on time relevance
MATCH (entity:Entity {name: $entity_name})
MATCH (entity)-[r]-(related:Entity)
WHERE r.timestamp > datetime() - duration({days: 30})
RETURN related.name, type(r) as relationship, r.timestamp
ORDER BY r.timestamp DESC
LIMIT 15
```

**Community-based Context Retrieval**:
```cypher
// Retrieve context from the same community
MATCH (start:Entity {name: $entity_name})
MATCH (start)-[:BELONGS_TO]->(community:Community)
MATCH (community)<-[:BELONGS_TO]-(related:Entity)
WHERE related.name <> start.name
RETURN related.name, community.name as community
LIMIT 20
```

#### **Python Implementation for LLM Context Retrieval**:

**Basic Context Retrieval Class**:
```python
from neo4j import GraphDatabase
from typing import List, Dict, Any
import json

class CypherContextRetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_entity_context(self, entity_name: str, max_results: int = 10) -> List[Dict]:
        """Retrieve context for a specific entity"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (entity:Entity {name: $entity_name})
                MATCH (entity)-[r]-(related)
                RETURN entity.name as entity, 
                       type(r) as relationship, 
                       labels(related) as related_type,
                       related.name as related_name,
                       properties(related) as related_props
                LIMIT $max_results
                """,
                entity_name=entity_name, max_results=max_results
            )
            return [record.data() for record in result]
    
    def get_semantic_context(self, concept: str, max_distance: int = 3) -> List[Dict]:
        """Retrieve semantically related concepts"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (start:Concept {name: $concept})
                MATCH path = (start)-[*1..$max_distance]-(related:Concept)
                WHERE related.name <> start.name
                RETURN related.name, length(path) as distance,
                       [node in nodes(path) | node.name] as path_nodes
                ORDER BY distance
                LIMIT 20
                """,
                concept=concept, max_distance=max_distance
            )
            return [record.data() for record in result]
    
    def get_weighted_context(self, entity_name: str) -> List[Dict]:
        """Retrieve context with relationship weights"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (entity:Entity {name: $entity_name})
                MATCH (entity)-[r]-(related)
                WITH entity, related, r,
                     CASE type(r)
                       WHEN 'WORKS_FOR' THEN 10
                       WHEN 'KNOWS' THEN 5
                       WHEN 'LIVES_IN' THEN 3
                       ELSE 1
                     END as importance
                ORDER BY importance DESC
                RETURN related.name, type(r) as relationship, importance
                LIMIT 15
                """,
                entity_name=entity_name
            )
            return [record.data() for record in result]
    
    def format_context_for_llm(self, context_data: List[Dict]) -> str:
        """Format context data for LLM consumption"""
        formatted_context = []
        
        for item in context_data:
            if 'relationship' in item:
                formatted_context.append(
                    f"{item.get('entity', 'Unknown')} {item['relationship']} {item.get('related_name', 'Unknown')}"
                )
            elif 'distance' in item:
                formatted_context.append(
                    f"Related to {item['name']} (distance: {item['distance']})"
                )
        
        return "\n".join(formatted_context)
```

**Advanced Context Retrieval with Caching**:
```python
import redis
import hashlib
import json
from typing import Optional

class AdvancedCypherContextRetriever(CypherContextRetriever):
    def __init__(self, uri: str, user: str, password: str, redis_url: str = None):
        super().__init__(uri, user, password)
        self.redis_client = redis.Redis.from_url(redis_url) if redis_url else None
        self.cache_ttl = 300  # 5 minutes
    
    def _get_cache_key(self, query_type: str, params: Dict) -> str:
        """Generate cache key for query"""
        key_data = f"{query_type}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached result if available"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Cache read error: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict]):
        """Cache result with TTL"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(result)
            )
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def get_entity_context_cached(self, entity_name: str, max_results: int = 10) -> List[Dict]:
        """Retrieve context with caching"""
        cache_key = self._get_cache_key("entity_context", {
            "entity_name": entity_name,
            "max_results": max_results
        })
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Get fresh data
        result = self.get_entity_context(entity_name, max_results)
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
```

#### **LLM Integration with Cypher Context Retrieval**:

**LangChain Integration**:
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class LLMWithGraphContext:
    def __init__(self, openai_api_key: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.llm = OpenAI(api_key=openai_api_key, temperature=0.7)
        self.context_retriever = CypherContextRetriever(neo4j_uri, neo4j_user, neo4j_password)
        
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Based on the following context from a knowledge graph, answer the question.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def answer_with_context(self, question: str, entity_name: str = None) -> str:
        """Answer question using graph context"""
        # Extract entity from question if not provided
        if not entity_name:
            entity_name = self._extract_entity_from_question(question)
        
        # Retrieve context
        context_data = self.context_retriever.get_entity_context(entity_name)
        context = self.context_retriever.format_context_for_llm(context_data)
        
        # Generate answer with context
        response = self.chain.run(question=question, context=context)
        return response
    
    def _extract_entity_from_question(self, question: str) -> str:
        """Simple entity extraction - in production, use NER"""
        # This is a simplified version - use proper NER in production
        words = question.split()
        for word in words:
            if word[0].isupper():
                return word
        return "Unknown"
```

**Advanced LLM Integration with Multiple Context Sources**:
```python
class AdvancedLLMWithGraphContext:
    def __init__(self, openai_api_key: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.llm = OpenAI(api_key=openai_api_key, temperature=0.7)
        self.context_retriever = AdvancedCypherContextRetriever(
            neo4j_uri, neo4j_user, neo4j_password
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["question", "entity_context", "semantic_context", "weighted_context"],
            template="""
            Answer the question using the following context from a knowledge graph:
            
            Entity Context (direct relationships):
            {entity_context}
            
            Semantic Context (related concepts):
            {semantic_context}
            
            Weighted Context (importance-based):
            {weighted_context}
            
            Question: {question}
            
            Provide a comprehensive answer based on the graph context:"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def answer_with_comprehensive_context(self, question: str, entity_name: str) -> str:
        """Answer with multiple types of context"""
        # Get different types of context
        entity_context = self.context_retriever.get_entity_context(entity_name)
        semantic_context = self.context_retriever.get_semantic_context(entity_name)
        weighted_context = self.context_retriever.get_weighted_context(entity_name)
        
        # Format contexts
        entity_context_str = self.context_retriever.format_context_for_llm(entity_context)
        semantic_context_str = self.context_retriever.format_context_for_llm(semantic_context)
        weighted_context_str = self.context_retriever.format_context_for_llm(weighted_context)
        
        # Generate comprehensive answer
        response = self.chain.run(
            question=question,
            entity_context=entity_context_str,
            semantic_context=semantic_context_str,
            weighted_context=weighted_context_str
        )
        
        return response
```

#### **Performance Optimization for Context Retrieval**:

**Query Optimization Techniques**:
```cypher
// Create indexes for better performance
CREATE INDEX entity_name_index FOR (e:Entity) ON (e.name)
CREATE INDEX concept_name_index FOR (c:Concept) ON (c.name)
CREATE INDEX relationship_type_index FOR ()-[r]-() ON (type(r))

// Use parameterized queries for better caching
MATCH (entity:Entity {name: $entity_name})
MATCH (entity)-[r]-(related)
WHERE type(r) IN $relationship_types
RETURN entity.name, type(r), related.name
LIMIT $max_results

// Use graph algorithms for complex context
CALL gds.pageRank.stream('myGraph', {
    nodeLabels: ['Entity'],
    relationshipTypes: ['RELATES_TO']
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name as entity, score
ORDER BY score DESC
LIMIT 20
```

**Caching Strategies**:
```python
class OptimizedContextRetriever:
    def __init__(self, neo4j_uri: str, redis_url: str):
        self.driver = GraphDatabase.driver(neo4j_uri)
        self.redis_client = redis.Redis.from_url(redis_url)
        self.query_cache = {}
    
    def get_context_with_caching(self, entity_name: str, query_type: str = "basic"):
        """Get context with intelligent caching"""
        cache_key = f"context:{entity_name}:{query_type}"
        
        # Check Redis cache first
        cached_result = self._get_redis_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Check in-memory cache
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Execute query
        result = self._execute_context_query(entity_name, query_type)
        
        # Cache results
        self._set_redis_cache(cache_key, result)
        self.query_cache[cache_key] = result
        
        return result
    
    def _execute_context_query(self, entity_name: str, query_type: str):
        """Execute optimized context query"""
        with self.driver.session() as session:
            if query_type == "basic":
                query = """
                MATCH (entity:Entity {name: $entity_name})
                MATCH (entity)-[r]-(related)
                RETURN entity.name, type(r), related.name
                LIMIT 10
                """
            elif query_type == "semantic":
                query = """
                MATCH (entity:Entity {name: $entity_name})
                MATCH path = (entity)-[*1..3]-(related)
                RETURN related.name, length(path) as distance
                ORDER BY distance
                LIMIT 15
                """
            
            result = session.run(query, entity_name=entity_name)
            return [record.data() for record in result]
```

#### **Best Practices for Cypher Context Retrieval**:

**Query Design**:
- **Use Indexes**: Create indexes on frequently queried properties
- **Limit Results**: Use LIMIT to prevent excessive data retrieval
- **Parameterized Queries**: Use parameters for better caching
- **Relationship Types**: Filter by specific relationship types when possible

**Performance Optimization**:
- **Caching**: Implement Redis caching for frequently accessed context
- **Query Optimization**: Use EXPLAIN to analyze query performance
- **Connection Pooling**: Reuse database connections
- **Batch Processing**: Process multiple entities in batches

**Context Quality**:
- **Relevance Scoring**: Implement relevance scoring for context
- **Freshness**: Consider temporal aspects of context
- **Diversity**: Ensure context covers different aspects
- **Validation**: Validate context quality before sending to LLM

### Graph Database Best Practices

#### **Data Modeling**:
- **Node Design**: Clear entity identification
- **Relationship Types**: Meaningful connection labels
- **Property Strategy**: Efficient property storage
- **Indexing**: Optimize query performance

#### **Performance Optimization**:
- **Query Optimization**: Efficient Cypher queries
- **Indexing**: Create appropriate indexes
- **Partitioning**: Distribute data across clusters
- **Caching**: Implement query result caching

#### **Scalability**:
- **Horizontal Scaling**: Add more database instances
- **Read Replicas**: Distribute read load
- **Sharding**: Partition data across nodes
- **Backup and Recovery**: Regular backups and testing

## Interview Preparation Tips

### **Key Concepts to Master**:
1. **Docker**: Containerization, Dockerfile optimization, orchestration
2. **Git/GitHub**: Version control, collaboration workflows, branching strategies
3. **GitHub Actions**: CI/CD pipelines, workflow automation, deployment
4. **pytest**: Testing frameworks, fixtures, parameterization, mocking
5. **APIs**: REST design, authentication, testing, documentation
6. **GraphDB**: Graph modeling, query optimization, use cases

### **Common Interview Questions with Detailed Answers**:

#### **1. How would you containerize a Python application with Docker?**

**Answer**: Follow a systematic approach to containerization:

**Dockerfile Design**:
```dockerfile
# Multi-stage build for optimization
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Security: non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "app.py"]
```

**Optimization Strategies**:
- **Layer Caching**: Copy requirements first for better caching
- **Multi-stage Builds**: Reduce final image size
- **Security**: Run as non-root user
- **Minimal Base Images**: Use slim or alpine images

**Production Considerations**:
- **Health Checks**: Monitor application health
- **Resource Limits**: Set memory and CPU limits
- **Logging**: Configure proper logging
- **Environment Variables**: Use for configuration

#### **2. How do you implement CI/CD with GitHub Actions?**

**Answer**: Create comprehensive CI/CD pipelines:

**Basic CI Pipeline**:
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8
    
    - name: Lint with flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: pytest --cov=app --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

**Advanced CD Pipeline**:
```yaml
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t myapp:${{ github.sha }} .
    
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        echo "Deploying to staging"
    
    - name: Run integration tests
      run: |
        # Run integration tests against staging
        echo "Running integration tests"
    
    - name: Deploy to production
      if: success()
      run: |
        # Deploy to production
        echo "Deploying to production"
```

**Best Practices**:
- **Security**: Use secrets for sensitive data
- **Caching**: Cache dependencies and build artifacts
- **Parallel Jobs**: Run independent jobs in parallel
- **Environment Management**: Separate staging and production

#### **3. How do you write effective tests with pytest?**

**Answer**: Follow comprehensive testing strategies:

**Test Organization**:
```python
# test_structure.py
import pytest
from unittest.mock import Mock, patch

class TestUserService:
    @pytest.fixture
    def user_service(self):
        return UserService()
    
    @pytest.fixture
    def mock_database(self):
        return Mock()
    
    def test_create_user_success(self, user_service, mock_database):
        with patch('app.database', mock_database):
            user_data = {"name": "John", "email": "john@example.com"}
            result = user_service.create_user(user_data)
            
            assert result["name"] == "John"
            assert result["email"] == "john@example.com"
            mock_database.insert.assert_called_once()
    
    @pytest.mark.parametrize("user_data,expected_error", [
        ({}, "Name is required"),
        ({"name": "John"}, "Email is required"),
        ({"name": "John", "email": "invalid"}, "Invalid email format"),
    ])
    def test_create_user_validation(self, user_service, user_data, expected_error):
        with pytest.raises(ValueError, match=expected_error):
            user_service.create_user(user_data)
```

**Advanced Testing Features**:
- **Fixtures**: Reusable test setup and teardown
- **Parameterization**: Test multiple scenarios efficiently
- **Mocking**: Isolate units under test
- **Coverage**: Ensure comprehensive test coverage

**Testing Best Practices**:
- **Test Isolation**: Each test should be independent
- **Descriptive Names**: Clear test method names
- **Arrange-Act-Assert**: Structure tests clearly
- **Edge Cases**: Test boundary conditions and errors

#### **4. How do you design a RESTful API?**

**Answer**: Follow REST principles and best practices:

**API Design Principles**:
```python
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from marshmallow import Schema, fields

app = Flask(__name__)
api = Api(app)

# Schema validation
class UserSchema(Schema):
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    age = fields.Int(validate=lambda n: 0 <= n <= 150)

class UserResource(Resource):
    def get(self, user_id=None):
        """GET /api/users or GET /api/users/{id}"""
        if user_id:
            user = get_user(user_id)
            if not user:
                return {"error": "User not found"}, 404
            return user, 200
        else:
            users = get_all_users()
            return {"users": users, "count": len(users)}, 200
    
    def post(self):
        """POST /api/users"""
        try:
            data = UserSchema().load(request.get_json())
            user = create_user(data)
            return user, 201
        except ValidationError as e:
            return {"errors": e.messages}, 400
    
    def put(self, user_id):
        """PUT /api/users/{id}"""
        try:
            data = UserSchema().load(request.get_json())
            user = update_user(user_id, data)
            if not user:
                return {"error": "User not found"}, 404
            return user, 200
        except ValidationError as e:
            return {"errors": e.messages}, 400
    
    def delete(self, user_id):
        """DELETE /api/users/{id}"""
        if delete_user(user_id):
            return "", 204
        return {"error": "User not found"}, 404

api.add_resource(UserResource, '/api/users', '/api/users/<int:user_id>')
```

**API Best Practices**:
- **Consistent Naming**: Use plural nouns for resources
- **Proper Status Codes**: Return appropriate HTTP status codes
- **Error Handling**: Provide meaningful error messages
- **Versioning**: Include API versioning strategy
- **Documentation**: Use OpenAPI/Swagger for documentation
- **Authentication**: Implement proper authentication and authorization

#### **5. How do you optimize a GraphDB query for performance?**

**Answer**: Implement comprehensive optimization strategies:

**Query Optimization**:
```cypher
// Before: Inefficient query
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
WHERE p.name = 'John Doe'
RETURN c.name

// After: Optimized with index
CREATE INDEX person_name_index FOR (p:Person) ON (p.name)
MATCH (p:Person {name: 'John Doe'})-[:WORKS_FOR]->(c:Company)
RETURN c.name

// Complex optimization
MATCH (start:Person {name: 'John Doe'})
CALL gds.shortestPath.dijkstra.stream('myGraph', {
    sourceNode: id(start),
    targetNode: id(end),
    relationshipWeightProperty: 'weight'
})
YIELD index, sourceNode, targetNode, totalCost, costs, path
RETURN path
```

**Performance Strategies**:
- **Indexing**: Create indexes on frequently queried properties
- **Query Planning**: Use EXPLAIN to analyze query plans
- **Caching**: Implement result caching for repeated queries
- **Partitioning**: Distribute data across multiple nodes

**Graph Database Best Practices**:
- **Data Modeling**: Design efficient graph schemas
- **Relationship Types**: Use meaningful relationship labels
- **Property Strategy**: Store properties efficiently
- **Query Patterns**: Use appropriate traversal patterns

This comprehensive guide covers all aspects of modern development tools and technologies, providing both theoretical understanding and practical implementation knowledge for your interview preparation. 