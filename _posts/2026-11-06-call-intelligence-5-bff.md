---
layout: post
title: "Serving the Dashboard: A Backend-for-Frontend for Call Intelligence"
summary: "How a small FastAPI service sits between the React dashboard and BigQuery, so the screens get the answer they need in one call and the browser never touches the warehouse."
project: call-intelligence
date: 2026-11-06 09:00:00
---

*Part 5 of a 5-part series on Call Intelligence. Parts 1 to 4 built the data. This part is how the dashboard reads it.*

> All call IDs, member details, and numbers in this article are **mock data** created for illustration.

---

## The data is built for analysis, not for screens

Parts 1 to 4 end with a set of BigQuery tables: one row per call in the consolidated dashboard table, plus per-job tables for summaries, topics, entities, resolution, and metrics. That layout is right for the pipeline. It is not the shape a screen wants.

A call detail page needs one object: the call's details, its summary, its topics, and its outcome. A KPI card needs one number. Asking the browser to assemble those from warehouse tables would mean the UI knows table names, joins, and partition rules. Every time a table changed, the frontend would break.

We put a **Backend-for-Frontend (BFF)** between them. It is a small service built for this one dashboard. It takes a screen's question, reads what it needs from BigQuery, and returns a response shaped for that screen.

---

## The shape of the system

<figure class="diagram">
{% include diagrams/bff-architecture.svg %}
<figcaption>Figure 1: The React dashboard calls an Apigee gateway, which checks the user's OAuth2 token. The FastAPI BFF validates the request, queries BigQuery, and returns a screen-ready response.</figcaption>
</figure>

Four pieces, each with one job:

- **React** draws the screens. It knows about endpoints and JSON, not tables.
- **Apigee** is the front door. It checks the user's OAuth2 token before a request reaches our code.
- **The FastAPI BFF** validates the request, runs the queries, and shapes the result.
- **BigQuery** stores and computes the data. Only the BFF has permission to read it.

The browser never holds BigQuery credentials. If someone steals a session token, they can call the API within its permissions, but they cannot read the warehouse directly.

---

## Two kinds of questions

Most dashboard requests fall into two groups, and the BFF reads each from the right place.

| Question | Example screen | Where it reads |
|---|---|---|
| **Across many calls** | KPI cards, trend charts, top call reasons | The consolidated dashboard table (one row per call, partitioned by `call_date`, clustered by line of business) |
| **About one call** | Call detail page | The per-job tables, joined on `call_id` |

The consolidated table answers "how many" and "which ones" quickly, because the query only touches the partitions and clusters the filter needs. The per-job tables hold the full detail for a single call, so the detail page can show everything without making the consolidated table wider than a dashboard needs.

<figure class="diagram">
{% include diagrams/bff-call-detail.svg %}
<figcaption>Figure 2: One request for a call ID reads five tables with that ID. The BFF merges the rows into a single call detail object.</figcaption>
</figure>

---

## The endpoints

The API is organized around screens, not tables. Each endpoint returns what one screen needs.

| Endpoint | What the screen needs | Returns |
|---|---|---|
| `GET /api/dashboard/metrics` | KPI cards | Volume, handle time, resolution rate, for a date range |
| `GET /api/calls?page=1&pageSize=50` | Call list | One page of calls with reason, date, and resolution |
| `GET /api/calls/{callId}` | Call detail page | Details, summary, topics, entities, and outcome in one object |
| `GET /api/topics/trends` | Trend chart | Topic share by day, from pre-aggregated rows |

A detail response looks like this (mock data):

```json
{
  "callId": "CALL-000123",
  "callDate": "2026-10-20",
  "lineOfBusiness": "Commercial",
  "summary": "Member asked why a prescription claim was denied and wanted to appeal.",
  "topics": [{ "name": "Ask why a claim was denied", "category": "Claims" }],
  "entities": [{ "type": "Benefit", "value": "Prescription coverage" }],
  "resolved": false,
  "durationSeconds": 542
}
```

The response is for the screen. The table layout behind it can change without the frontend noticing.

---

## Inside one request

The FastAPI service follows the same five steps for every request:

1. **Validate.** FastAPI checks the path and query parameters against a schema. A bad page size fails here, before any query runs.
2. **Decide the source.** Lists and KPIs go to the consolidated table. Detail goes to the per-job tables.
3. **Query.** The service runs a parameterized query. Values from the URL are passed as parameters, never pasted into the SQL text.
4. **Shape.** Rows are merged and renamed into the response model the screen expects.
5. **Respond or fail.** A success returns the object. A failure returns a clean error with no SQL or stack trace.

```python
@app.get("/api/calls/{call_id}", response_model=CallDetail)
async def get_call(call_id: str) -> CallDetail:
    rows = run_query(CALL_DETAIL_SQL, {"call_id": call_id})  # parameterized, see below
    if not rows:
        raise HTTPException(status_code=404, detail="CALL_NOT_FOUND")
    return to_call_detail(rows)
```

The FastAPI response model is also the contract. If the BFF returns a field the screen does not expect, the test suite fails before it reaches production.

---

## Keeping it fast

BigQuery is built for large scans. Dashboards need answers in well under a second. Four habits close that gap:

- **Pre-aggregate the KPIs.** Daily metrics are computed by the pipeline, so a KPI card reads a few rows, not millions of calls.
- **Filter on the partition.** Every list and trend query includes a `call_date` range, so BigQuery reads only the days it needs.
- **Select only what the screen uses.** No `SELECT *`. A call list does not need the summary text.
- **Page everything.** Lists return 50 rows at a time. The browser asks for the next page when the user scrolls.

---

## Security at the edge and in the service

Two checks happen on every request, and they answer different questions.

- **Apigee asks: who is calling?** It validates the OAuth2 access token and rejects requests without a valid one.
- **The BFF asks: what may this service read?** It runs as a service account with read-only access to the Call Intelligence datasets, and nothing else.

Keeping them separate means a valid user token does not give the service broader access, and a valid service account does not let anonymous traffic in.

---

## Errors and observability

Because every request passes through one service, the BFF is the right place to standardize errors and logs. The frontend gets the same error shape every time:

```json
{
  "error": {
    "code": "CALL_NOT_FOUND",
    "message": "The requested call could not be found."
  }
}
```

Each request gets a correlation ID, logged with its endpoint, status, total latency, and BigQuery time. That makes one question easy to answer: is a slow dashboard a frontend problem, a gateway problem, or a slow query?

---

## Deploying the BFF

The service is a container running on Google Kubernetes Engine (GKE). Deployment is automated with GitHub Actions, so every release follows the same path.

<figure class="diagram">
{% include diagrams/bff-deploy.svg %}
<figcaption>Figure 3: A push to main triggers the build and tests, publishes a versioned image, and rolls it out to GKE.</figcaption>
</figure>

The same image runs in every environment. Only configuration changes, so a release that passed in test is the one that runs in production.

---

## Lessons learned

1. **Design the API around the screen.** The first endpoints mirrored the tables. They were awkward to use, and each frontend change meant a backend change. Screen-shaped endpoints fixed that.
2. **Split the reads by question.** Lists and KPIs and single-call detail have different access patterns. One table for everything made both slower.
3. **Keep the warehouse behind the service.** Giving the browser direct access would have been faster to build and much harder to secure or change later.

---

## Where the series ends

Parts 1 to 4 explained how each call becomes structured data. This part explains how that data reaches a screen: a BFF that reads the consolidated table for overviews, reads the per-job tables for one call, and sits behind an Apigee gateway with OAuth2.

The data pipeline and the dashboard can now change independently. The BFF is the seam between them.
