# Retrieval API & Embedding Plan

## REST Contract (Planner-Facing)
- `GET /catalog/activities`
  - Query params: `board`, `specialty`, `modality[]`, `format[]`, `city`, `state`, `country`, `start_after`, `end_before`, `credit_type[]`, `min_credits`, `max_price`, `requirement_code[]`, `committed_only`, `cursor`, `limit`.
  - Response: envelope with `results`, `next_cursor`, `retrieval_context` (filters echoed + total hits).
  - Each activity includes core fields (`activity_id`, `provider_id`, `title`, `summary`, `start_date`, `end_date`, `credits`, `pricing_preview`, `eligibility_flags`, `last_verified_at`, `confidence_score`).
  - `pricing_preview` surfaces the cheapest tier and tier count; full breakdown available via detail endpoint.
- `GET /catalog/activities/{activity_id}`
  - Returns normalized record: activity core, sessions, pricing tiers, credit types, eligibility requirements, commitment, requirement mappings, provenance.
  - Optional `?include=documents,quality_signals`.
- `POST /catalog/search`
  - Body: `{ "query": "...", "filters": { ... }, "max_results": 30 }`.
  - Invokes hybrid retrieval (structured filter + vector similarity). Response includes `results` with `score_breakdown` (structure vs semantic) and `explanations` describing why each candidate surfaced.
- `GET /catalog/providers/{provider_id}`
  - Provider profile, catalog counts, update cadence, ingestion health metrics.

All responses carry `retrieved_at` timestamp and `staleness_seconds` so the planner can prefer fresh data or trigger re-ingest.

## GraphQL Schema Sketch (UI / Assistant)
```graphql
type Query {
  activities(
    filter: ActivityFilter
    after: String
    first: Int = 25
  ): ActivityConnection!

  activity(id: ID!): Activity

  providers(filter: ProviderFilter, after: String, first: Int = 25): ProviderConnection!
}

input ActivityFilter {
  board: String
  requirementCodes: [String!]
  modalities: [Modality!]
  formats: [ActivityFormat!]
  creditTypes: [CreditType!]
  minCredits: Float
  maxPrice: Float
  startAfter: Date
  endBefore: Date
  allowHybrid: Boolean
  onlyCommitted: Boolean
  topicIds: [ID!]
  location: LocationInput
  search: String
}

type Activity {
  id: ID!
  provider: Provider!
  title: String!
  slug: String!
  summary: String
  modality: Modality!
  format: ActivityFormat!
  startDate: Date
  endDate: Date
  city: String
  state: String
  country: String
  credits: [CreditBreakdown!]!
  pricing: [PricingTier!]!
  eligibility: [EligibilityRequirement!]!
  commitment: CommitmentDetail
  requirementMappings: [RequirementMapping!]!
  provenance: Provenance!
  confidenceScore: Float!
  fieldGaps: [String!]!
}
```

GraphQL wrappers expose the same underlying catalog tables and let the assistant request only the slices it needs (e.g., pricing tiers + sessions when the user asks for scheduling details).

## Embedding Strategy
- **Vector shape**: store text embeddings per activity version, session (optional), provider summary, and requirement knowledge base. Each vector row retains `embedding_model`, `embedding_version`, `source_table`, `source_pk`, `last_embedded_at`.
- **Models**: default to `text-embedding-3-large` (or successor) for long-form summaries; maintain fallback to `text-embedding-3-small` for cost-sensitive refreshes. For requirement documents, use the same model so cosine similarity is comparable.
- **Content mix**: create one "dense" document per activity combining title, summary, learning objectives, credit notes, eligibility, and pricing highlights. Keep length <= 2k tokens. Additional mini-docs for provider reputation and user feedback feed ranking features.
- **Refresh cadence**: re-embed on `activity.updated_at` or when enrichment jobs fill previously missing fields (eligibility, pricing). Batch embeddings nightly with change feed; allow on-demand refresh for urgent corrections.
- **Hybrid scoring**: retrieval service first applies structured filters, then runs kNN search on embeddings for relevance. Candidate scoring weights: 0.6 semantic, 0.3 requirement overlap (categorical matching), 0.1 recency. Provide `score_breakdown` in API responses for transparency.
- **Caching**: store embeddings in pgvector or a dedicated vector DB (e.g., Milvus, Pinecone) depending on infra strategy. Include cache invalidation hooks when activities are archived or providers pause offerings.

## Web Search Fallback Triggering
- Planner monitors `field_gaps`, `confidence_score`, and `staleness_seconds`. If below thresholds or catalog returns < `min_results`, issue a secondary discovery request through Perplexity/Google/OpenAI.
- Fallback pipeline logs candidate URLs back into the ingestion queue with priority tags so the crawler can attempt structured extraction and re-run embeddings automatically.
- Responses to the user clearly mark fallback results as "freshly sourced" with lower confidence until enrichment is complete.
