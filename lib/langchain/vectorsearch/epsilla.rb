# frozen_string_literal: true

module Langchain::Vectorsearch
  class Epsilla < Base
    #
    # Wrapper around Epsilla client library
    #
    # Gem requirements:
    #     gem "epsilla-ruby", "~> 0.0.3"
    #
    # Usage:
    #     epsilla = Langchain::Vectorsearch::Epsilla.new(protocol:, host:, port:, db_name:, db_path:, index_name:, llm:)
    #
    # Initialize Epsilla client
    # @param protocol [String] The protocol to use, e.g. http or https
    # @param host [String] The host to connect to
    # @param db_name [String] The name of the database to use
    # @param db_path [String] The path to the database to use
    # @param index_name [String] The name of the Epsilla table to use
    # @param llm [Object] The LLM client to use
    # @param port [Integer] The port to connect to, default 8888
    def initialize(protocol:, host:, db_name:, db_path:, index_name:, llm:, port: 8888)
      depends_on "epsilla-ruby", req: "epsilla"

      @client = Epsilla::Client.new(
        protocol: protocol,
        host: host,
        port: port
      )

      status_code, response = @client.database.load_db(db_name: db_name, db_path: db_path)
      raise "Failed to load database: #{response}" if status_code != 200

      @db_name = db_name
      @db_path = db_path
      @table_name = index_name

      @vector_dimension = llm.default_dimension

      super(llm: llm)
    end

    # Method supported by Vectorsearch DB to retrieve a default schema
    # TODO: fix me
    def get_default_schema
      @client.database.list_tables
    end

    # Create a table using the index_name passed in the constructor
    def create_default_schema
      status_code, response = @client.database.create_table(
        table_name: @table_name,
        table_fields: [
          {"name" => "ID", "dataType" => "INT", "primaryKey" => true},
          {"name" => "Doc", "dataType" => "STRING"},
          {"name" => "Embedding", "dataType" => "VECTOR_FLOAT", "dimensions" => @vector_dimension}
        ]
      )
      raise "Failed to create table: #{response}" if status_code != 200
      response
    end

    # Drop the table using the index_name passed in the constructor
    def destroy_default_schema
      @client.database.drop_table(@table_name)
    end

    # Add a list of texts to the database
    # @param texts [Array<String>] The list of texts to add
    def add_texts(texts:)
      data = texts.map do |text, idx|
        {Doc: text, Embedding: llm.embed(text: text).embedding, ID: idx}
      end

      status_code, response = @client.database.insert(
        table_name: @table_name,
        table_records: data
      )
      raise "Failed to insert texts: #{response}" if status_code != 200
      response
    end

    # Search for similar texts
    # @param query [String] The text to search for
    # @param k [Integer] The number of results to return
    # @return [String] The response from the server
    def similarity_search(query:, k: 4)
      embedding = llm.embed(text: query).embedding

      similarity_search_by_vector(
        embedding: embedding,
        k: k
      )
    end

    # Search for entries by embedding
    # @param embedding [Array<Float>] The embedding to search for
    # @param k [Integer] The number of results to return
    # @return [String] The response from the server
    def similarity_search_by_vector(embedding:, k: 4)
      status_code, response = @client.database.query(
        table_name: @table_name,
        query_field: "Embedding",
        query_vector: embedding,
        response_fields: ["Doc"],
        limit: k,
        with_distance: false
      )
      raise "Failed to do similarity search: #{response}" if status_code != 200

      response["result"]
    end

    # Ask a question and return the answer
    # @param question [String] The question to ask
    # @param k [Integer] The number of results to have in context
    # @yield [String] Stream responses back one String at a time
    # @return [String] The answer to the question
    def ask(question:, k: 4, &block)
      search_results = similarity_search(query: question, k: k)

      context = search_results.map do |result|
        result.content.to_s
      end
      context = context.join("\n---\n")

      prompt = generate_rag_prompt(question: question, context: context)

      response = llm.chat(prompt: prompt, &block)
      response.context = context
      response
    end
  end
end
