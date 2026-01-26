import requests
import base64

class ShopifyClient:
    def __init__(self, store_url, access_token):
        self.store_url = store_url.rstrip('/')
        self.access_token = access_token
        self.api_version = '2024-01'
        self.base_url = f"https://{self.store_url}/admin/api/{self.api_version}"
        self.headers = {
            'X-Shopify-Access-Token': access_token,
            'Content-Type': 'application/json'
        }
    
    def find_product_by_sku(self, sku):
        """
        Find a product by its SKU (stored in variant).
        Returns the product and variant IDs if found.
        """
        # Search for variant with this SKU
        url = f"{self.base_url}/variants.json"
        params = {'sku': sku}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Shopify API error: {response.status_code} - {response.text}")
        
        variants = response.json().get('variants', [])
        
        if not variants:
            return None, None
        
        variant = variants[0]
        product_id = variant['product_id']
        variant_id = variant['id']
        
        return product_id, variant_id
    
    def get_product(self, product_id):
        """Get a product by ID."""
        url = f"{self.base_url}/products/{product_id}.json"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Shopify API error: {response.status_code} - {response.text}")
        
        return response.json().get('product')
    
    def add_image_to_product(self, product_id, image_url, position=None, alt_text=None):
        """
        Add an image to a product from a URL.
        """
        url = f"{self.base_url}/products/{product_id}/images.json"
        
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
        product_id, variant_id = self.find_product_by_sku(sku)
        
        if not product_id:
            raise Exception(f"No product found with SKU: {sku}")
        
        product = self.get_product(product_id)
        
        # Get current image count to set positions
        current_image_count = len(product.get('images', []))
        
        added_images = []
        for i, image_url in enumerate(image_urls):
            position = current_image_count + i + 1
            image = self.add_image_to_product(
                product_id, 
                image_url, 
                position=position,
                alt_text=f"{product.get('title', '')} - Image {position}"
            )
            added_images.append(image)
        
        return {
            'product_id': product_id,
            'product_title': product.get('title'),
            'sku': sku,
            'images_added': len(added_images)
        }
