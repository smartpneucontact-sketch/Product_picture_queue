import requests

class ShopifyClient:
    def __init__(self, store_url, access_token):
        self.store_url = store_url.rstrip('/')
        self.access_token = access_token
        self.api_version = '2024-10'
        self.graphql_url = f"https://{self.store_url}/admin/api/{self.api_version}/graphql.json"
        self.rest_url = f"https://{self.store_url}/admin/api/{self.api_version}"
        self.headers = {
            'X-Shopify-Access-Token': access_token,
            'Content-Type': 'application/json'
        }
    
    def _graphql(self, query, variables=None):
        """Execute a GraphQL query."""
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        response = requests.post(self.graphql_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Shopify GraphQL error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if 'errors' in result:
            raise Exception(f"Shopify GraphQL error: {result['errors']}")
        
        return result.get('data')
    
    def find_product_by_sku(self, sku):
        """
        Find a product by its SKU using GraphQL.
        Returns (product_id, variant_id) as numeric IDs.
        """
        query = """
        query findProductBySku($query: String!) {
            productVariants(first: 1, query: $query) {
                edges {
                    node {
                        id
                        sku
                        product {
                            id
                            title
                        }
                    }
                }
            }
        }
        """
        
        data = self._graphql(query, {'query': f'sku:{sku}'})
        edges = data.get('productVariants', {}).get('edges', [])
        
        if not edges:
            return None, None, None
        
        node = edges[0]['node']
        
        # Extract numeric IDs from GID format (gid://shopify/Product/123456)
        product_gid = node['product']['id']
        variant_gid = node['id']
        product_id = product_gid.split('/')[-1]
        variant_id = variant_gid.split('/')[-1]
        product_title = node['product']['title']
        
        return product_id, variant_id, product_title
    
    def get_product_images_count(self, product_id):
        """Get current number of images on a product."""
        query = """
        query getProductImages($id: ID!) {
            product(id: $id) {
                images(first: 250) {
                    edges {
                        node {
                            id
                        }
                    }
                }
            }
        }
        """
        
        product_gid = f"gid://shopify/Product/{product_id}"
        data = self._graphql(query, {'id': product_gid})
        
        if not data.get('product'):
            return 0
        
        return len(data['product']['images']['edges'])
    
    def add_image_to_product(self, product_id, image_url, position=None, alt_text=None):
        """
        Add an image to a product using REST API (more reliable for image uploads).
        """
        url = f"{self.rest_url}/products/{product_id}/images.json"
        
        payload = {
            'image': {
                'src': image_url
            }
        }
        
        if position is not None:
            payload['image']['position'] = position
        
        if alt_text:
            payload['image']['alt'] = alt_text
        
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Shopify API error: {response.status_code} - {response.text}")
        
        return response.json().get('image')
    
    def add_images_to_product_by_sku(self, sku, image_urls):
        """
        Add multiple images to a product identified by SKU.
        Returns the product info and added images.
        """
        product_id, variant_id, product_title = self.find_product_by_sku(sku)
        
        if not product_id:
            raise Exception(f"No product found with SKU: {sku}")
        
        # Get current image count to set positions
        current_image_count = self.get_product_images_count(product_id)
        
        added_images = []
        errors = []
        
        for i, image_url in enumerate(image_urls):
            try:
                position = current_image_count + i + 1
                image = self.add_image_to_product(
                    product_id, 
                    image_url, 
                    position=position,
                    alt_text=f"{product_title} - Image {position}"
                )
                added_images.append(image)
            except Exception as e:
                errors.append(f"Image {i+1}: {str(e)}")
        
        if errors and not added_images:
            raise Exception(f"All image uploads failed: {'; '.join(errors)}")
        
        return {
            'product_id': product_id,
            'product_title': product_title,
            'sku': sku,
            'images_added': len(added_images),
            'errors': errors if errors else None
        }
