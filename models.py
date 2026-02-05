from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Image(db.Model):
    __tablename__ = 'images'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Original image from camera
    original_filename = db.Column(db.String(255), nullable=False)
    original_url = db.Column(db.String(500), nullable=False)
    
    # Processed image (after bg removal + crop)
    processed_url = db.Column(db.String(500), nullable=True)
    
    # SKU assignment
    sku = db.Column(db.String(100), nullable=True, index=True)
    
    # Image type: front (default), side, gauge
    image_type = db.Column(db.String(20), default='front')
    
    # Status: pending, assigned, processing, completed, failed
    status = db.Column(db.String(20), default='pending', index=True)
    error_message = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    assigned_at = db.Column(db.DateTime, nullable=True)
    processed_at = db.Column(db.DateTime, nullable=True)
    
    # Position in queue (for manual reordering if needed)
    position = db.Column(db.Integer, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'original_filename': self.original_filename,
            'original_url': self.original_url,
            'processed_url': self.processed_url,
            'sku': self.sku,
            'image_type': self.image_type or 'front',
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
        }
