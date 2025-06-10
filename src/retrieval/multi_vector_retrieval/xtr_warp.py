from typing import List, Optional, Union

import requests


class XTRWarpClient:
    """
    A client-side search implementation that communicates with a remote search server.
    This class mimics the interface of CustomSearcher but delegates the actual search
    operations to a remote server via HTTP requests.
    """

    def __init__(
        self,
        address: str = "http://localhost",
        port: int = 7000,
        topk: int = 10,
        return_as_text: bool = False,
    ):
        """
        Initialize the RemoteSearcher with server URL.

        Args:
            address: Address of the search server
            port: Port of the search server
            topk: Number of results to return
            return_as_text: If True, returns decoded text chunks instead of IDs
        """
        self.server_url = f"{address}:{port}".rstrip("/")
        self.topk = topk
        self.return_as_text = return_as_text

    def search(
        self, query: str, k: Optional[int] = None, return_as_text: Optional[bool] = None
    ) -> Union[List[int], List[str]]:
        """
        Search for the top-k results matching the given query by sending a request to the server.

        Args:
            query: The search query string
            k: Number of results to return (default: None)
            return_as_text: If True, returns decoded text chunks instead of IDs (default: None)

        Returns:
            Either a list of global chunk IDs or a list of text chunks,
            depending on the return_as_text parameter
        """
        # Set the topk and return_as_text if they are not provided
        if k is None:
            k = self.topk
        if return_as_text is None:
            return_as_text = self.return_as_text

        # Prepare the request payload
        payload = {"query": query, "k": k, "return_as_text": return_as_text}

        # Send the request to the server
        response = requests.post(f"{self.server_url}/search", json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data["results"]
        else:
            # Handle error cases
            error_message = response.json().get("error", "Unknown error")
            raise Exception(f"Search request failed: {error_message}")

    def search_multiple(
        self,
        queries: List[str],
        k: Optional[int] = None,
        return_as_text: Optional[bool] = None,
    ) -> Union[List[List[int]], List[List[str]]]:
        """
        Perform search for multiple queries by sending a request to the search_multiple endpoint.

        Args:
            queries: List of query strings to search for
            k: Number of results to return per query (default: None)
            return_as_text: If True, returns decoded text chunks instead of IDs (default: None)

        Returns:
            Either a list of lists of global chunk IDs or a list of lists of text chunks
        """
        # Set the topk and return_as_text if they are not provided
        if k is None:
            k = self.topk
        if return_as_text is None:
            return_as_text = self.return_as_text

        # Prepare the request payload
        payload = {"queries": queries, "k": k, "return_as_text": return_as_text}

        try:
            # Send the request to the search_multiple endpoint
            response = requests.post(f"{self.server_url}/search_multiple", json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                return data["results"]
            else:
                # Handle error cases
                error_message = response.json().get("error", "Unknown error")
                raise Exception(f"Search multiple request failed: {error_message}")
        except Exception as e:
            # Fall back to individual searches if the endpoint fails
            print(
                f"Warning: search_multiple endpoint failed, falling back to individual searches: {str(e)}"
            )
            results = []
            for query in queries:
                result = self.search(query, k, return_as_text)
                results.append(result)
            return results

    def search_batch(
        self,
        queries: List[str],
        k: Optional[int] = None,
        return_as_text: Optional[bool] = None,
    ) -> Union[List[List[int]], List[List[str]]]:
        """
        Perform batch search for multiple queries at once by sending a request to the search_batch endpoint.

        Args:
            queries: List of query strings to search for
            k: Number of results to return per query (default: None)
            return_as_text: If True, returns decoded text chunks instead of IDs (default: None)

        Returns:
            Either a list of lists of global chunk IDs or a list of lists of text chunks
        """
        # Set the topk and return_as_text if they are not provided
        if k is None:
            k = self.topk
        if return_as_text is None:
            return_as_text = self.return_as_text

        # Prepare the request payload
        payload = {"queries": queries, "k": k, "return_as_text": return_as_text}

        try:
            # Send the request to the search_batch endpoint
            response = requests.post(f"{self.server_url}/search_batch", json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                return data["results"]
            else:
                # Fall back to search_multiple if batch endpoint returns an error
                error_message = response.json().get("error", "Unknown error")
                print(
                    f"Warning: search_batch endpoint failed, falling back to search_multiple: {error_message}"
                )
                return self.search_multiple(queries, k, return_as_text)
        except Exception as e:
            # Fall back to search_multiple if batch endpoint doesn't exist or fails
            print(
                f"Warning: search_batch endpoint failed, falling back to search_multiple: {str(e)}"
            )
            return self.search_multiple(queries, k, return_as_text)

    def convert_global_chunk_id_to_text(self, global_chunk_id: int) -> str:
        """
        Convert a global chunk ID to its corresponding text by sending a request to the server.

        Args:
            global_chunk_id: The global ID of the chunk to convert

        Returns:
            The decoded text of the specified chunk
        """
        # Prepare the request payload
        payload = {"global_chunk_id": global_chunk_id}

        # Send the request to the server
        response = requests.post(f"{self.server_url}/convert_to_text", json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data["text"]
        else:
            # Handle error cases
            error_message = response.json().get("error", "Unknown error")
            raise Exception(f"Text conversion request failed: {error_message}")


if __name__ == "__main__":
    searcher = Retriever(
        address="http://localhost", port=5000, topk=3, return_as_text=False
    )

    # Test single search
    print("Testing single search...")
    results = searcher.search(
        "When did Barack Obama become president?", k=3, return_as_text=True
    )
    print(f"Single search results: {results}\n")

    # Test search_multiple
    print("Testing search_multiple...")
    queries = ["When did Barack Obama become president?", "How do butterflies taste?"]
    results = searcher.search_multiple(queries, k=3, return_as_text=True)
    print(f"Search multiple results: {results}\n")

    # Test search_batch
    print("Testing search_batch...")
    results = searcher.search_batch(queries, k=3, return_as_text=True)
    print(f"Search batch results: {results}\n")

    # Test single search with return_as_text=False
    print("Testing single search with return_as_text=False...")
    results = searcher.search(
        "When did Barack Obama become president?", k=3, return_as_text=False
    )
    print(f"Single search results: {results}\n")

    # Test convert_global_chunk_id_to_text
    print("Testing convert_global_chunk_id_to_text...")
    text = searcher.convert_global_chunk_id_to_text(results[0])
    print(f"Converted text: {text}\n")
