import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import hashlib
from typing import Dict, List, Tuple, Optional
import uuid
import time
import base64
from io import BytesIO
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# For PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Optional: Import for real Claude AI integration
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Enterprise Database Migration Tool",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DatabaseSizingEngine:
    """AI-powered database sizing and configuration engine"""
    
    def __init__(self):
        self.instance_specs = {
            # RDS Instance Types
            "db.t3.micro": {"vcpu": 2, "memory": 1, "network": "Low", "cost_factor": 1.0, "use_case": "Dev/Test"},
            "db.t3.small": {"vcpu": 2, "memory": 2, "network": "Low to Moderate", "cost_factor": 1.5, "use_case": "Dev/Test"},
            "db.t3.medium": {"vcpu": 2, "memory": 4, "network": "Low to Moderate", "cost_factor": 2.2, "use_case": "Small Production"},
            "db.t3.large": {"vcpu": 2, "memory": 8, "network": "Low to Moderate", "cost_factor": 3.8, "use_case": "Small Production"},
            "db.m5.large": {"vcpu": 2, "memory": 8, "network": "Up to 10 Gbps", "cost_factor": 4.2, "use_case": "General Purpose"},
            "db.m5.xlarge": {"vcpu": 4, "memory": 16, "network": "Up to 10 Gbps", "cost_factor": 8.4, "use_case": "General Purpose"},
            "db.m5.2xlarge": {"vcpu": 8, "memory": 32, "network": "Up to 10 Gbps", "cost_factor": 16.8, "use_case": "General Purpose"},
            "db.m5.4xlarge": {"vcpu": 16, "memory": 64, "network": "Up to 10 Gbps", "cost_factor": 33.6, "use_case": "High Performance"},
            "db.m5.8xlarge": {"vcpu": 32, "memory": 128, "network": "10 Gbps", "cost_factor": 67.2, "use_case": "High Performance"},
            "db.m5.12xlarge": {"vcpu": 48, "memory": 192, "network": "12 Gbps", "cost_factor": 100.8, "use_case": "High Performance"},
            "db.m5.16xlarge": {"vcpu": 64, "memory": 256, "network": "20 Gbps", "cost_factor": 134.4, "use_case": "High Performance"},
            "db.m5.24xlarge": {"vcpu": 96, "memory": 384, "network": "25 Gbps", "cost_factor": 201.6, "use_case": "High Performance"},
            "db.r5.large": {"vcpu": 2, "memory": 16, "network": "Up to 10 Gbps", "cost_factor": 5.5, "use_case": "Memory Optimized"},
            "db.r5.xlarge": {"vcpu": 4, "memory": 32, "network": "Up to 10 Gbps", "cost_factor": 11.0, "use_case": "Memory Optimized"},
            "db.r5.2xlarge": {"vcpu": 8, "memory": 64, "network": "Up to 10 Gbps", "cost_factor": 22.0, "use_case": "Memory Optimized"},
            "db.r5.4xlarge": {"vcpu": 16, "memory": 128, "network": "Up to 10 Gbps", "cost_factor": 44.0, "use_case": "Memory Optimized"},
            "db.r5.8xlarge": {"vcpu": 32, "memory": 256, "network": "10 Gbps", "cost_factor": 88.0, "use_case": "Memory Optimized"},
            "db.r5.12xlarge": {"vcpu": 48, "memory": 384, "network": "12 Gbps", "cost_factor": 132.0, "use_case": "Memory Optimized"},
            "db.r5.16xlarge": {"vcpu": 64, "memory": 512, "network": "20 Gbps", "cost_factor": 176.0, "use_case": "Memory Optimized"},
            "db.r5.24xlarge": {"vcpu": 96, "memory": 768, "network": "25 Gbps", "cost_factor": 264.0, "use_case": "Memory Optimized"},
            # Aurora Instance Types
            "db.r5.large": {"vcpu": 2, "memory": 16, "network": "Up to 10 Gbps", "cost_factor": 5.5, "use_case": "Aurora"},
            "db.r5.xlarge": {"vcpu": 4, "memory": 32, "network": "Up to 10 Gbps", "cost_factor": 11.0, "use_case": "Aurora"},
            "db.r5.2xlarge": {"vcpu": 8, "memory": 64, "network": "Up to 10 Gbps", "cost_factor": 22.0, "use_case": "Aurora"},
            "db.r5.4xlarge": {"vcpu": 16, "memory": 128, "network": "Up to 10 Gbps", "cost_factor": 44.0, "use_case": "Aurora"},
            "db.r5.8xlarge": {"vcpu": 32, "memory": 256, "network": "10 Gbps", "cost_factor": 88.0, "use_case": "Aurora"},
            "db.r5.12xlarge": {"vcpu": 48, "memory": 384, "network": "12 Gbps", "cost_factor": 132.0, "use_case": "Aurora"},
            "db.r5.16xlarge": {"vcpu": 64, "memory": 512, "network": "20 Gbps", "cost_factor": 176.0, "use_case": "Aurora"},
            "db.r5.24xlarge": {"vcpu": 96, "memory": 768, "network": "25 Gbps", "cost_factor": 264.0, "use_case": "Aurora"}
        }
        
        self.workload_patterns = {
            "OLTP": {"cpu_weight": 0.3, "memory_weight": 0.4, "io_weight": 0.3, "read_write_ratio": 0.7},
            "OLAP": {"cpu_weight": 0.4, "memory_weight": 0.3, "io_weight": 0.3, "read_write_ratio": 0.9},
            "Mixed": {"cpu_weight": 0.35, "memory_weight": 0.35, "io_weight": 0.3, "read_write_ratio": 0.8},
            "Analytics": {"cpu_weight": 0.5, "memory_weight": 0.3, "io_weight": 0.2, "read_write_ratio": 0.95},
            "Reporting": {"cpu_weight": 0.25, "memory_weight": 0.45, "io_weight": 0.3, "read_write_ratio": 0.85}
        }
        
        self.environment_factors = {
            "Development": {"scaling_factor": 0.3, "availability": "Single-AZ", "backup_retention": 7},
            "QA": {"scaling_factor": 0.5, "availability": "Single-AZ", "backup_retention": 7},
            "SQA": {"scaling_factor": 0.7, "availability": "Multi-AZ", "backup_retention": 14},
            "Production": {"scaling_factor": 1.0, "availability": "Multi-AZ", "backup_retention": 30}
        }
    
    def analyze_vrops_data(self, vrops_data):
        """Analyze vROps data for sizing recommendations"""
        if not vrops_data:
            return None
        
        # Simulate vROps analysis - in real implementation, this would parse actual vROps metrics
        analysis = {
            "peak_cpu": vrops_data.get("cpu_peak", 0),
            "avg_cpu": vrops_data.get("cpu_avg", 0),
            "peak_memory": vrops_data.get("memory_peak", 0),
            "avg_memory": vrops_data.get("memory_avg", 0),
            "peak_iops": vrops_data.get("iops_peak", 0),
            "avg_iops": vrops_data.get("iops_avg", 0),
            "growth_trend": vrops_data.get("growth_rate", 0.15)
        }
        
        return analysis
    
    def calculate_sizing_requirements(self, config, vrops_data=None, use_ai=True):
        """Calculate optimal database sizing based on workload and requirements"""
        
        # Get base requirements
        workload_type = config.get("workload_type", "Mixed")
        environment = config.get("environment", "Production")
        database_size_gb = config.get("database_size_gb", 100)
        concurrent_connections = config.get("concurrent_connections", 100)
        transactions_per_second = config.get("transactions_per_second", 1000)
        growth_rate = config.get("annual_growth_rate", 0.2)
        
        # Analyze vROps data if available
        if vrops_data:
            vrops_analysis = self.analyze_vrops_data(vrops_data)
            if vrops_analysis:
                # Adjust requirements based on vROps data
                cpu_requirement = max(vrops_analysis["peak_cpu"] * 1.2, 2)  # 20% headroom
                memory_requirement = max(vrops_analysis["peak_memory"] * 1.3, 4)  # 30% headroom
                iops_requirement = max(vrops_analysis["peak_iops"] * 1.1, 1000)  # 10% headroom
            else:
                # Fallback to algorithmic calculation
                cpu_requirement, memory_requirement, iops_requirement = self._calculate_algorithmic_sizing(config)
        else:
            # Use algorithmic calculation
            cpu_requirement, memory_requirement, iops_requirement = self._calculate_algorithmic_sizing(config)
        
        # Apply environment factors
        env_factor = self.environment_factors[environment]["scaling_factor"]
        cpu_requirement *= env_factor
        memory_requirement *= env_factor
        iops_requirement *= env_factor
        
        # Find optimal instance types
        writer_instance = self._find_optimal_instance(cpu_requirement, memory_requirement, "writer")
        reader_instances = self._calculate_reader_requirements(config, writer_instance)
        
        # Calculate storage requirements
        storage_config = self._calculate_storage_requirements(database_size_gb, iops_requirement, environment)
        
        # Growth projections
        growth_projections = self._calculate_growth_projections(
            writer_instance, reader_instances, storage_config, growth_rate
        )
        
        return {
            "writer_instance": writer_instance,
            "reader_instances": reader_instances,
            "storage_config": storage_config,
            "sizing_rationale": self._generate_sizing_rationale(config, cpu_requirement, memory_requirement),
            "growth_projections": growth_projections,
            "environment_config": self.environment_factors[environment]
        }
    
    def _calculate_algorithmic_sizing(self, config):
        """Fallback algorithmic sizing when vROps data is not available"""
        
        workload_type = config.get("workload_type", "Mixed")
        database_size_gb = config.get("database_size_gb", 100)
        concurrent_connections = config.get("concurrent_connections", 100)
        transactions_per_second = config.get("transactions_per_second", 1000)
        
        workload_pattern = self.workload_patterns[workload_type]
        
        # CPU calculation based on TPS and connections
        base_cpu = (transactions_per_second / 1000) * 2  # 2 vCPU per 1000 TPS baseline
        connection_cpu = (concurrent_connections / 100) * 1  # 1 vCPU per 100 connections
        cpu_requirement = max(2, base_cpu + connection_cpu) * workload_pattern["cpu_weight"] * 2
        
        # Memory calculation based on database size and connections
        base_memory = max(4, database_size_gb * 0.1)  # 10% of DB size as baseline
        connection_memory = (concurrent_connections * 10) / 1024  # 10MB per connection
        memory_requirement = (base_memory + connection_memory) * workload_pattern["memory_weight"] * 2
        
        # IOPS calculation based on TPS and workload pattern
        base_iops = transactions_per_second * 5  # 5 IOPS per transaction baseline
        iops_requirement = base_iops * workload_pattern["io_weight"] * 2
        
        return cpu_requirement, memory_requirement, iops_requirement
    
    def _find_optimal_instance(self, cpu_req, memory_req, instance_role):
        """Find the optimal instance type based on requirements"""
        
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in self.instance_specs.items():
            if specs["vcpu"] >= cpu_req and specs["memory"] >= memory_req:
                # Calculate efficiency score (lower is better)
                cpu_efficiency = specs["vcpu"] / cpu_req
                memory_efficiency = specs["memory"] / memory_req
                cost_efficiency = specs["cost_factor"]
                
                # Combined efficiency score
                score = (cpu_efficiency + memory_efficiency) * cost_efficiency
                
                if score < best_score:
                    best_score = score
                    best_instance = {
                        "instance_type": instance_type,
                        "specs": specs,
                        "efficiency_score": score,
                        "cpu_utilization": (cpu_req / specs["vcpu"]) * 100,
                        "memory_utilization": (memory_req / specs["memory"]) * 100
                    }
        
        return best_instance or {
            "instance_type": "db.m5.large",
            "specs": self.instance_specs["db.m5.large"],
            "efficiency_score": 1.0,
            "cpu_utilization": 50,
            "memory_utilization": 50
        }
    
    def _calculate_reader_requirements(self, config, writer_instance):
        """Calculate optimal number and type of read replicas"""
        
        workload_type = config.get("workload_type", "Mixed")
        read_query_percentage = config.get("read_query_percentage", 70)
        high_availability_required = config.get("high_availability", True)
        
        workload_pattern = self.workload_patterns[workload_type]
        read_write_ratio = workload_pattern["read_write_ratio"]
        
        # Calculate number of readers needed
        if read_query_percentage >= 80:
            num_readers = 3 if high_availability_required else 2
        elif read_query_percentage >= 60:
            num_readers = 2 if high_availability_required else 1
        else:
            num_readers = 1 if high_availability_required else 0
        
        # Reader instances can be smaller than writer for read-heavy workloads
        reader_instance_type = writer_instance["instance_type"]
        if read_query_percentage >= 80 and workload_type in ["OLAP", "Analytics", "Reporting"]:
            # Use same size for read-heavy analytical workloads
            reader_specs = writer_instance["specs"]
        else:
            # Use smaller instances for general read replicas
            reader_specs = self._get_smaller_instance(writer_instance["instance_type"])
            reader_instance_type = reader_specs["instance_type"]
        
        readers = []
        for i in range(num_readers):
            readers.append({
                "instance_type": reader_instance_type,
                "specs": reader_specs["specs"] if isinstance(reader_specs, dict) and "specs" in reader_specs else reader_specs,
                "role": f"read-replica-{i+1}",
                "cross_az": True if high_availability_required else False
            })
        
        return readers
    
    def _get_smaller_instance(self, current_instance):
        """Get a smaller instance type for read replicas"""
        
        instance_hierarchy = [
            "db.t3.micro", "db.t3.small", "db.t3.medium", "db.t3.large",
            "db.m5.large", "db.m5.xlarge", "db.m5.2xlarge", "db.m5.4xlarge",
            "db.m5.8xlarge", "db.m5.12xlarge", "db.m5.16xlarge", "db.m5.24xlarge"
        ]
        
        try:
            current_index = instance_hierarchy.index(current_instance)
            smaller_index = max(0, current_index - 1)
            smaller_instance = instance_hierarchy[smaller_index]
            
            return {
                "instance_type": smaller_instance,
                "specs": self.instance_specs[smaller_instance]
            }
        except ValueError:
            return {
                "instance_type": "db.m5.large",
                "specs": self.instance_specs["db.m5.large"]
            }
    
    def _calculate_storage_requirements(self, database_size_gb, iops_requirement, environment):
        """Calculate optimal storage configuration"""
        
        # Add growth buffer
        total_storage_gb = database_size_gb * 1.5  # 50% growth buffer
        
        # Determine storage type based on IOPS requirements
        if iops_requirement > 32000:
            storage_type = "io2"
            provisioned_iops = iops_requirement
        elif iops_requirement > 16000:
            storage_type = "io1"
            provisioned_iops = iops_requirement
        elif iops_requirement > 3000:
            storage_type = "gp3"
            provisioned_iops = min(16000, iops_requirement)
        else:
            storage_type = "gp2"
            provisioned_iops = min(3000, max(100, total_storage_gb * 3))  # GP2 baseline
        
        # Backup and snapshot configuration
        env_config = self.environment_factors[environment]
        
        return {
            "storage_type": storage_type,
            "allocated_storage_gb": int(total_storage_gb),
            "provisioned_iops": int(provisioned_iops),
            "backup_retention_days": env_config["backup_retention"],
            "multi_az": env_config["availability"] == "Multi-AZ",
            "encryption_enabled": True,
            "snapshot_frequency": "daily" if environment == "Production" else "weekly"
        }
    
    def _calculate_growth_projections(self, writer_instance, reader_instances, storage_config, growth_rate):
        """Calculate growth projections for capacity planning"""
        
        projections = {}
        
        for year in [1, 2, 3]:
            growth_factor = (1 + growth_rate) ** year
            
            # Storage growth
            projected_storage = storage_config["allocated_storage_gb"] * growth_factor
            
            # Compute growth (simplified - may need instance upgrades)
            if growth_factor > 2.0:  # Significant growth
                compute_scaling = "Consider instance upgrades or additional read replicas"
            elif growth_factor > 1.5:
                compute_scaling = "Monitor performance, may need scaling"
            else:
                compute_scaling = "Current configuration sufficient"
            
            projections[f"year_{year}"] = {
                "storage_gb": int(projected_storage),
                "growth_factor": growth_factor,
                "compute_recommendation": compute_scaling,
                "estimated_cost_increase": f"{((growth_factor - 1) * 100):.1f}%"
            }
        
        return projections
    
    def _generate_sizing_rationale(self, config, cpu_req, memory_req):
        """Generate human-readable rationale for sizing decisions"""
        
        workload_type = config.get("workload_type", "Mixed")
        environment = config.get("environment", "Production")
        
        rationale = f"""
        Sizing analysis for {workload_type} workload in {environment} environment:
        
        • CPU Requirements: {cpu_req:.1f} vCPUs based on transaction volume and workload pattern
        • Memory Requirements: {memory_req:.1f} GB based on database size and connection patterns
        • Environment Factor: {self.environment_factors[environment]['scaling_factor']} applied for {environment}
        • Availability: {self.environment_factors[environment]['availability']} configuration
        """
        
        return rationale.strip()

class DatabaseMigrationCostCalculator:
    """Calculate comprehensive database migration costs with real-time AWS pricing"""
    
    def __init__(self):
        self.pricing_client = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        self._init_pricing_client()
        
        # Base pricing for fallback (USD per hour/month)
        self.fallback_pricing = {
            "rds": {
                "db.t3.micro": 0.017,
                "db.t3.small": 0.034,
                "db.t3.medium": 0.068,
                "db.t3.large": 0.136,
                "db.m5.large": 0.192,
                "db.m5.xlarge": 0.384,
                "db.m5.2xlarge": 0.768,
                "db.m5.4xlarge": 1.536,
                "db.m5.8xlarge": 3.072,
                "db.m5.12xlarge": 4.608,
                "db.m5.16xlarge": 6.144,
                "db.m5.24xlarge": 9.216,
                "db.r5.large": 0.252,
                "db.r5.xlarge": 0.504,
                "db.r5.2xlarge": 1.008,
                "db.r5.4xlarge": 2.016,
                "db.r5.8xlarge": 4.032,
                "db.r5.12xlarge": 6.048,
                "db.r5.16xlarge": 8.064,
                "db.r5.24xlarge": 12.096
            },
            "aurora": {
                "db.r5.large": 0.29,
                "db.r5.xlarge": 0.58,
                "db.r5.2xlarge": 1.16,
                "db.r5.4xlarge": 2.32,
                "db.r5.8xlarge": 4.64,
                "db.r5.12xlarge": 6.96,
                "db.r5.16xlarge": 9.28,
                "db.r5.24xlarge": 13.92
            },
            "storage": {
                "gp2": 0.115,  # per GB per month
                "gp3": 0.08,   # per GB per month
                "io1": 0.125,  # per GB per month
                "io2": 0.125   # per GB per month
            },
            "iops": {
                "io1": 0.065,  # per IOPS per month
                "io2": 0.065   # per IOPS per month
            }
        }
    
    def _init_pricing_client(self):
        """Initialize AWS pricing client"""
        try:
            if hasattr(st, 'secrets') and 'aws' in st.secrets:
                self.pricing_client = boto3.client(
                    'pricing',
                    region_name='us-east-1',  # Pricing API only available in us-east-1
                    aws_access_key_id=st.secrets["aws"]["access_key_id"],
                    aws_secret_access_key=st.secrets["aws"]["secret_access_key"]
                )
            else:
                self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        except Exception as e:
            st.warning(f"Could not initialize AWS pricing client: {str(e)}")
            self.pricing_client = None
    
    def get_rds_pricing(self, instance_type, region="us-east-1", engine="postgres"):
        """Get real-time RDS pricing"""
        if not self.pricing_client:
            return self.fallback_pricing["rds"].get(instance_type, 0.192)
        
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonRDS',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': engine},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_location_name(region)},
                    {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                terms = price_data['terms']['OnDemand']
                
                for term_key, term_value in terms.items():
                    for price_dimension_key, price_dimension in term_value['priceDimensions'].items():
                        if 'USD' in price_dimension['pricePerUnit']:
                            return float(price_dimension['pricePerUnit']['USD'])
        except Exception as e:
            st.warning(f"Error fetching RDS pricing: {str(e)}")
        
        return self.fallback_pricing["rds"].get(instance_type, 0.192)
    
    def get_aurora_pricing(self, instance_type, region="us-east-1"):
        """Get real-time Aurora pricing"""
        if not self.pricing_client:
            return self.fallback_pricing["aurora"].get(instance_type, 0.29)
        
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonRDS',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': 'Aurora PostgreSQL'},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_location_name(region)}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                terms = price_data['terms']['OnDemand']
                
                for term_key, term_value in terms.items():
                    for price_dimension_key, price_dimension in term_value['priceDimensions'].items():
                        if 'USD' in price_dimension['pricePerUnit']:
                            return float(price_dimension['pricePerUnit']['USD'])
        except Exception as e:
            st.warning(f"Error fetching Aurora pricing: {str(e)}")
        
        return self.fallback_pricing["aurora"].get(instance_type, 0.29)
    
    def _get_location_name(self, region):
        """Map AWS region codes to location names"""
        location_mapping = {
            'us-east-1': 'US East (N. Virginia)',
            'us-east-2': 'US East (Ohio)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'eu-central-1': 'Europe (Frankfurt)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)'
        }
        return location_mapping.get(region, 'US East (N. Virginia)')
    
    def calculate_total_migration_cost(self, sizing_config, migration_config, growth_projections):
        """Calculate comprehensive migration costs"""
        
        # Migration-specific costs
        migration_costs = self._calculate_migration_costs(migration_config)
        
        # Ongoing operational costs
        operational_costs = self._calculate_operational_costs(sizing_config, migration_config)
        
        # Growth-based costs
        growth_costs = self._calculate_growth_costs(growth_projections, operational_costs)
        
        # Network and data transfer costs
        network_costs = self._calculate_network_costs(migration_config)
        
        total_monthly = (
            operational_costs['monthly_total'] + 
            network_costs['monthly_total']
        )
        
        total_yearly = total_monthly * 12
        
        return {
            "migration_costs": migration_costs,
            "operational_costs": operational_costs,
            "network_costs": network_costs,
            "growth_costs": growth_costs,
            "summary": {
                "total_monthly": total_monthly,
                "total_yearly": total_yearly,
                "migration_one_time": migration_costs['total_migration_cost']
            }
        }
    
    def _calculate_migration_costs(self, migration_config):
        """Calculate one-time migration costs"""
        
        database_size_gb = migration_config.get('database_size_gb', 100)
        migration_method = migration_config.get('migration_method', 'DMS')
        downtime_tolerance = migration_config.get('downtime_tolerance', 'Low')
        
        # DMS costs
        if migration_method == 'DMS':
            dms_instance_cost = 0.5 * 24 * 30  # $0.5/hour for dms.t3.medium for 30 days
            dms_data_transfer = database_size_gb * 0.02  # $0.02 per GB
            dms_total = dms_instance_cost + dms_data_transfer
        else:
            dms_total = 0
        
        # Professional services
        if downtime_tolerance == 'Zero':
            professional_services = 15000  # Complex zero-downtime migration
        elif downtime_tolerance == 'Low':
            professional_services = 8000   # Standard migration with minimal downtime
        else:
            professional_services = 3000   # Simple migration with acceptable downtime
        
        # Testing and validation
        testing_costs = 2000  # Standard testing costs
        
        # Training
        training_costs = 1500
        
        total_migration_cost = dms_total + professional_services + testing_costs + training_costs
        
        return {
            "dms_costs": dms_total,
            "professional_services": professional_services,
            "testing_costs": testing_costs,
            "training_costs": training_costs,
            "total_migration_cost": total_migration_cost
        }
    
    def _calculate_operational_costs(self, sizing_config, migration_config):
        """Calculate ongoing operational costs"""
        
        service_type = migration_config.get('target_service', 'RDS')
        region = migration_config.get('target_region', 'us-east-1')
        
        # Compute costs
        writer_cost = self._get_instance_cost(
            sizing_config['writer_instance']['instance_type'], 
            service_type, 
            region
        )
        
        reader_costs = 0
        for reader in sizing_config['reader_instances']:
            reader_costs += self._get_instance_cost(
                reader['instance_type'], 
                service_type, 
                region
            )
        
        # Storage costs
        storage_config = sizing_config['storage_config']
        storage_gb = storage_config['allocated_storage_gb']
        storage_type = storage_config['storage_type']
        
        storage_cost_per_gb = self.fallback_pricing['storage'][storage_type]
        monthly_storage_cost = storage_gb * storage_cost_per_gb
        
        # IOPS costs (for io1/io2)
        iops_cost = 0
        if storage_type in ['io1', 'io2']:
            provisioned_iops = storage_config['provisioned_iops']
            iops_rate = self.fallback_pricing['iops'][storage_type]
            iops_cost = provisioned_iops * iops_rate
        
        # Backup costs
        backup_cost = storage_gb * 0.095  # $0.095 per GB per month for backups
        
        # Multi-AZ multiplier
        multi_az_multiplier = 2.0 if storage_config['multi_az'] else 1.0
        
        # Calculate totals
        compute_monthly = (writer_cost + reader_costs) * 24 * 30 * multi_az_multiplier
        storage_monthly = monthly_storage_cost + iops_cost + backup_cost
        monthly_total = compute_monthly + storage_monthly
        
        return {
            "compute_monthly": compute_monthly,
            "storage_monthly": storage_monthly,
            "backup_monthly": backup_cost,
            "monthly_total": monthly_total,
            "breakdown": {
                "writer_instance": writer_cost * 24 * 30,
                "reader_instances": reader_costs * 24 * 30,
                "storage": monthly_storage_cost,
                "iops": iops_cost,
                "backups": backup_cost,
                "multi_az_factor": multi_az_multiplier
            }
        }
    
    def _get_instance_cost(self, instance_type, service_type, region):
        """Get instance cost per hour"""
        if service_type == 'Aurora':
            return self.get_aurora_pricing(instance_type, region)
        else:
            return self.get_rds_pricing(instance_type, region)
    
    def _calculate_network_costs(self, migration_config):
        """Calculate network and connectivity costs"""
        
        connectivity_type = migration_config.get('connectivity_type', 'VPN')
        database_size_gb = migration_config.get('database_size_gb', 100)
        
        # Data transfer costs
        data_transfer_cost = database_size_gb * 0.09  # $0.09 per GB out
        
        # Connectivity costs
        if connectivity_type == 'Direct Connect':
            dx_cost = 1000  # $1000/month for 1Gbps DX
        elif connectivity_type == 'VPN':
            dx_cost = 50    # $50/month for VPN
        else:
            dx_cost = 0     # Public internet
        
        # VPC Endpoint costs (if used)
        vpc_endpoint_cost = 30  # $30/month for VPC endpoints
        
        monthly_total = dx_cost + vpc_endpoint_cost
        
        return {
            "data_transfer_one_time": data_transfer_cost,
            "connectivity_monthly": dx_cost,
            "vpc_endpoints_monthly": vpc_endpoint_cost,
            "monthly_total": monthly_total
        }
    
    def _calculate_growth_costs(self, growth_projections, base_operational_costs):
        """Calculate costs with growth projections"""
        
        growth_costs = {}
        base_monthly = base_operational_costs['monthly_total']
        
        for year, projection in growth_projections.items():
            growth_factor = projection['growth_factor']
            projected_monthly = base_monthly * growth_factor
            growth_costs[year] = {
                "monthly_cost": projected_monthly,
                "yearly_cost": projected_monthly * 12,
                "growth_factor": growth_factor
            }
        
        return growth_costs

class DatabaseMigrationRiskAssessment:
    """Comprehensive risk assessment for database migrations"""
    
    def __init__(self):
        self.risk_factors = {
            "technical": {
                "data_size": {"low": 100, "medium": 1000, "high": 10000},  # GB
                "downtime_tolerance": {"zero": "high", "low": "medium", "medium": "low"},
                "complexity": {"simple": "low", "moderate": "medium", "complex": "high"},
                "heterogeneous": {"homogeneous": "low", "heterogeneous": "high"}
            },
            "business": {
                "criticality": {"low": "low", "medium": "medium", "high": "high", "critical": "critical"},
                "user_impact": {"minimal": "low", "moderate": "medium", "significant": "high"},
                "compliance": {"none": "low", "basic": "medium", "strict": "high"}
            },
            "operational": {
                "team_experience": {"expert": "low", "experienced": "medium", "novice": "high"},
                "rollback_plan": {"comprehensive": "low", "basic": "medium", "none": "high"},
                "testing_coverage": {"comprehensive": "low", "adequate": "medium", "minimal": "high"}
            }
        }
    
    def assess_migration_risks(self, config):
        """Perform comprehensive risk assessment"""
        
        risk_scores = {}
        
        # Technical risks
        technical_risks = self._assess_technical_risks(config)
        risk_scores["technical"] = technical_risks
        
        # Business risks
        business_risks = self._assess_business_risks(config)
        risk_scores["business"] = business_risks
        
        # Operational risks
        operational_risks = self._assess_operational_risks(config)
        risk_scores["operational"] = operational_risks
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(risk_scores)
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(risk_scores)
        
        return {
            "risk_scores": risk_scores,
            "overall_risk": overall_risk,
            "mitigation_strategies": mitigation_strategies,
            "risk_matrix": self._create_risk_matrix(risk_scores)
        }
    
    def _assess_technical_risks(self, config):
        """Assess technical migration risks"""
        
        database_size_gb = config.get('database_size_gb', 100)
        migration_type = config.get('migration_type', 'homogeneous')
        downtime_tolerance = config.get('downtime_tolerance', 'medium')
        schema_complexity = config.get('schema_complexity', 'moderate')
        
        risks = {}
        
        # Data size risk
        if database_size_gb > 10000:
            risks["data_size"] = {"level": "high", "impact": "Extended migration time, higher failure risk"}
        elif database_size_gb > 1000:
            risks["data_size"] = {"level": "medium", "impact": "Moderate migration time"}
        else:
            risks["data_size"] = {"level": "low", "impact": "Quick migration possible"}
        
        # Migration type risk
        if migration_type == "heterogeneous":
            risks["migration_type"] = {"level": "high", "impact": "Schema conversion required, compatibility issues"}
        else:
            risks["migration_type"] = {"level": "low", "impact": "Direct migration possible"}
        
        # Downtime tolerance risk
        downtime_risk_map = {"zero": "high", "low": "medium", "medium": "low", "high": "low"}
        risks["downtime"] = {
            "level": downtime_risk_map.get(downtime_tolerance, "medium"),
            "impact": "Business continuity requirements"
        }
        
        return risks
    
    def _assess_business_risks(self, config):
        """Assess business-related risks"""
        
        business_criticality = config.get('business_criticality', 'medium')
        compliance_requirements = config.get('compliance_frameworks', [])
        user_base_size = config.get('user_base_size', 'medium')
        
        risks = {}
        
        # Business criticality risk
        risks["criticality"] = {
            "level": business_criticality,
            "impact": f"Business impact during migration: {business_criticality}"
        }
        
        # Compliance risk
        if len(compliance_requirements) > 2:
            risks["compliance"] = {"level": "high", "impact": "Multiple compliance frameworks to maintain"}
        elif len(compliance_requirements) > 0:
            risks["compliance"] = {"level": "medium", "impact": "Compliance requirements must be validated"}
        else:
            risks["compliance"] = {"level": "low", "impact": "No specific compliance requirements"}
        
        # User impact risk
        user_impact_map = {"small": "low", "medium": "medium", "large": "high", "enterprise": "critical"}
        risks["user_impact"] = {
            "level": user_impact_map.get(user_base_size, "medium"),
            "impact": f"Impact on {user_base_size} user base"
        }
        
        return risks
    
    def _assess_operational_risks(self, config):
        """Assess operational risks"""
        
        team_experience = config.get('team_experience', 'experienced')
        rollback_plan = config.get('rollback_plan', 'basic')
        testing_strategy = config.get('testing_strategy', 'adequate')
        
        risks = {}
        
        # Team experience risk
        experience_risk_map = {"expert": "low", "experienced": "medium", "novice": "high"}
        risks["team_experience"] = {
            "level": experience_risk_map.get(team_experience, "medium"),
            "impact": "Team capability to handle migration complexity"
        }
        
        # Rollback plan risk
        rollback_risk_map = {"comprehensive": "low", "basic": "medium", "none": "high"}
        risks["rollback_plan"] = {
            "level": rollback_risk_map.get(rollback_plan, "medium"),
            "impact": "Ability to recover from migration failures"
        }
        
        # Testing coverage risk
        testing_risk_map = {"comprehensive": "low", "adequate": "medium", "minimal": "high"}
        risks["testing"] = {
            "level": testing_risk_map.get(testing_strategy, "medium"),
            "impact": "Confidence in migration success"
        }
        
        return risks
    
    def _calculate_overall_risk(self, risk_scores):
        """Calculate overall risk level"""
        
        risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        total_score = 0
        risk_count = 0
        
        for category, risks in risk_scores.items():
            for risk_name, risk_data in risks.items():
                level = risk_data["level"]
                total_score += risk_levels.get(level, 2)
                risk_count += 1
        
        if risk_count == 0:
            return "medium"
        
        average_score = total_score / risk_count
        
        if average_score <= 1.5:
            return "low"
        elif average_score <= 2.5:
            return "medium"
        elif average_score <= 3.5:
            return "high"
        else:
            return "critical"
    
    def _generate_mitigation_strategies(self, risk_scores):
        """Generate risk mitigation strategies"""
        
        strategies = {
            "technical": [],
            "business": [],
            "operational": []
        }
        
        # Technical mitigation strategies
        for risk_name, risk_data in risk_scores["technical"].items():
            if risk_data["level"] in ["high", "critical"]:
                if risk_name == "data_size":
                    strategies["technical"].append("Use parallel processing and data partitioning")
                    strategies["technical"].append("Implement incremental migration approach")
                elif risk_name == "migration_type":
                    strategies["technical"].append("Extensive schema conversion testing")
                    strategies["technical"].append("Use AWS Schema Conversion Tool")
                elif risk_name == "downtime":
                    strategies["technical"].append("Implement zero-downtime migration with DMS")
                    strategies["technical"].append("Use read replicas for cutover")
        
        # Business mitigation strategies
        for risk_name, risk_data in risk_scores["business"].items():
            if risk_data["level"] in ["high", "critical"]:
                if risk_name == "criticality":
                    strategies["business"].append("Develop comprehensive rollback procedures")
                    strategies["business"].append("Implement gradual user migration")
                elif risk_name == "compliance":
                    strategies["business"].append("Engage compliance team early")
                    strategies["business"].append("Conduct compliance validation testing")
        
        # Operational mitigation strategies
        for risk_name, risk_data in risk_scores["operational"].items():
            if risk_data["level"] in ["high", "critical"]:
                if risk_name == "team_experience":
                    strategies["operational"].append("Engage AWS Professional Services")
                    strategies["operational"].append("Provide team training before migration")
                elif risk_name == "rollback_plan":
                    strategies["operational"].append("Develop detailed rollback procedures")
                    strategies["operational"].append("Test rollback scenarios")
                elif risk_name == "testing":
                    strategies["operational"].append("Implement comprehensive testing strategy")
                    strategies["operational"].append("Conduct performance and load testing")
        
        return strategies
    
    def _create_risk_matrix(self, risk_scores):
        """Create risk matrix for visualization"""
        
        matrix = []
        
        for category, risks in risk_scores.items():
            for risk_name, risk_data in risks.items():
                matrix.append({
                    "category": category,
                    "risk": risk_name,
                    "level": risk_data["level"],
                    "impact": risk_data["impact"]
                })
        
        return matrix

class DatabaseMigrationPlanner:
    """Generate comprehensive migration plans"""
    
    def __init__(self):
        self.migration_phases = {
            "assessment": {
                "duration_weeks": 2,
                "activities": [
                    "Current state analysis",
                    "Performance baseline establishment", 
                    "Schema analysis and conversion planning",
                    "Risk assessment",
                    "Target architecture design"
                ]
            },
            "preparation": {
                "duration_weeks": 3,
                "activities": [
                    "AWS environment setup",
                    "Network connectivity establishment",
                    "Security configuration",
                    "Migration tools configuration",
                    "Test environment preparation"
                ]
            },
            "testing": {
                "duration_weeks": 4,
                "activities": [
                    "Schema conversion testing",
                    "Data migration testing",
                    "Application connectivity testing",
                    "Performance testing",
                    "Rollback procedure testing"
                ]
            },
            "migration": {
                "duration_weeks": 2,
                "activities": [
                    "Initial data sync",
                    "Incremental replication",
                    "Application cutover",
                    "DNS updates",
                    "Go-live validation"
                ]
            },
            "optimization": {
                "duration_weeks": 2,
                "activities": [
                    "Performance tuning",
                    "Cost optimization",
                    "Monitoring setup",
                    "Documentation",
                    "Team training"
                ]
            }
        }
    
    def generate_migration_plan(self, config, risk_assessment):
        """Generate comprehensive migration plan"""
        
        # Adjust timeline based on risk level
        risk_level = risk_assessment["overall_risk"]
        timeline_adjustment = self._get_timeline_adjustment(risk_level)
        
        # Generate timeline
        timeline = self._create_migration_timeline(timeline_adjustment)
        
        # Generate phase details
        phase_details = self._generate_phase_details(config, risk_assessment)
        
        # Generate resource requirements
        resource_requirements = self._generate_resource_requirements(config)
        
        # Generate success criteria
        success_criteria = self._generate_success_criteria(config)
        
        return {
            "timeline": timeline,
            "phase_details": phase_details,
            "resource_requirements": resource_requirements,
            "success_criteria": success_criteria,
            "risk_mitigation": risk_assessment["mitigation_strategies"]
        }
    
    def _get_timeline_adjustment(self, risk_level):
        """Adjust timeline based on risk level"""
        adjustments = {
            "low": 1.0,
            "medium": 1.2,
            "high": 1.5,
            "critical": 2.0
        }
        return adjustments.get(risk_level, 1.2)
    
    def _create_migration_timeline(self, adjustment_factor):
        """Create detailed migration timeline"""
        
        timeline = {}
        current_week = 0
        
        for phase_name, phase_info in self.migration_phases.items():
            adjusted_duration = int(phase_info["duration_weeks"] * adjustment_factor)
            
            timeline[phase_name] = {
                "start_week": current_week + 1,
                "end_week": current_week + adjusted_duration,
                "duration_weeks": adjusted_duration,
                "activities": phase_info["activities"]
            }
            
            current_week += adjusted_duration
        
        timeline["total_duration_weeks"] = current_week
        timeline["total_duration_months"] = round(current_week / 4.33, 1)
        
        return timeline
    
    def _generate_phase_details(self, config, risk_assessment):
        """Generate detailed phase information"""
        
        details = {}
        
        for phase_name in self.migration_phases.keys():
            details[phase_name] = {
                "objectives": self._get_phase_objectives(phase_name),
                "deliverables": self._get_phase_deliverables(phase_name),
                "risks": self._get_phase_risks(phase_name, risk_assessment),
                "success_criteria": self._get_phase_success_criteria(phase_name)
            }
        
        return details
    
    def _get_phase_objectives(self, phase_name):
        """Get objectives for each phase"""
        objectives = {
            "assessment": [
                "Understand current database architecture",
                "Identify migration requirements and constraints",
                "Design target AWS architecture"
            ],
            "preparation": [
                "Set up AWS target environment",
                "Configure migration tools and processes",
                "Establish network connectivity"
            ],
            "testing": [
                "Validate migration procedures",
                "Test application compatibility", 
                "Verify performance requirements"
            ],
            "migration": [
                "Execute production migration",
                "Minimize downtime and business impact",
                "Validate successful cutover"
            ],
            "optimization": [
                "Optimize performance and costs",
                "Complete documentation and training",
                "Establish ongoing operations"
            ]
        }
        return objectives.get(phase_name, [])
    
    def _get_phase_deliverables(self, phase_name):
        """Get deliverables for each phase"""
        deliverables = {
            "assessment": [
                "Current state assessment report",
                "Target architecture design",
                "Migration strategy document",
                "Risk assessment and mitigation plan"
            ],
            "preparation": [
                "AWS environment setup",
                "Migration tool configuration",
                "Network connectivity validation",
                "Security configuration documentation"
            ],
            "testing": [
                "Test migration results",
                "Performance test reports",
                "Application compatibility validation",
                "Rollback procedure verification"
            ],
            "migration": [
                "Production migration execution",
                "Cutover validation report",
                "Performance monitoring results",
                "Go-live sign-off"
            ],
            "optimization": [
                "Performance tuning results",
                "Cost optimization recommendations",
                "Operations documentation",
                "Team training completion"
            ]
        }
        return deliverables.get(phase_name, [])
    
    def _get_phase_risks(self, phase_name, risk_assessment):
        """Get phase-specific risks"""
        # This could be enhanced to map specific risks to phases
        return ["Phase-specific risks based on overall assessment"]
    
    def _get_phase_success_criteria(self, phase_name):
        """Get success criteria for each phase"""
        criteria = {
            "assessment": [
                "Complete understanding of current environment",
                "Approved target architecture design",
                "Signed-off migration plan"
            ],
            "preparation": [
                "AWS environment ready for migration",
                "All tools configured and tested",
                "Network connectivity validated"
            ],
            "testing": [
                "Successful test migrations completed",
                "Performance requirements validated",
                "Rollback procedures tested"
            ],
            "migration": [
                "Production migration completed successfully",
                "All applications functioning normally",
                "Performance within acceptable limits"
            ],
            "optimization": [
                "Performance optimized for production",
                "Operations procedures documented",
                "Team trained on new environment"
            ]
        }
        return criteria.get(phase_name, [])
    
    def _generate_resource_requirements(self, config):
        """Generate resource requirements for migration"""
        
        return {
            "team_composition": [
                "Database Migration Specialist",
                "AWS Solutions Architect", 
                "Application Developer",
                "Network Engineer",
                "Security Specialist",
                "Project Manager"
            ],
            "aws_services": [
                "Amazon RDS or Aurora",
                "AWS Database Migration Service",
                "AWS Schema Conversion Tool",
                "Amazon CloudWatch",
                "AWS VPC",
                "AWS Direct Connect (if needed)"
            ],
            "tools_and_software": [
                "Database migration tools",
                "Performance monitoring tools",
                "Backup and recovery tools",
                "Testing frameworks"
            ],
            "estimated_effort": {
                "total_person_weeks": 24,
                "key_roles": {
                    "Database Specialist": 8,
                    "AWS Architect": 6,
                    "Project Manager": 4,
                    "Other roles": 6
                }
            }
        }
    
    def _generate_success_criteria(self, config):
        """Generate overall success criteria"""
        
        return {
            "technical": [
                "Zero data loss during migration",
                "Application functionality maintained",
                "Performance meets or exceeds current levels",
                "All security requirements satisfied"
            ],
            "business": [
                "Minimal business disruption",
                "Migration completed within timeline",
                "Costs within approved budget",
                "Compliance requirements maintained"
            ],
            "operational": [
                "Team trained on new environment",
                "Documentation complete and accessible",
                "Monitoring and alerting operational",
                "Backup and recovery procedures tested"
            ]
        }

class DatabaseMigrationPlatform:
    """Main platform class orchestrating all database migration components"""
    
    def __init__(self):
        self.sizing_engine = DatabaseSizingEngine()
        self.cost_calculator = DatabaseMigrationCostCalculator()
        self.risk_assessor = DatabaseMigrationRiskAssessment()
        self.migration_planner = DatabaseMigrationPlanner()
        self.initialize_session_state()
        self.setup_custom_css()
    
    def initialize_session_state(self):
        """Initialize session state"""
        if 'migration_configs' not in st.session_state:
            st.session_state.migration_configs = {}
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "configuration"
    
    def setup_custom_css(self):
        """Setup custom CSS styling"""
        st.markdown("""
        <style>
            .main-header {
                background: linear-gradient(135deg, #FF9900 0%, #232F3E 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            
            .section-header {
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                margin: 1.5rem 0 1rem 0;
                font-size: 1.2rem;
                font-weight: bold;
                box-shadow: 0 2px 8px rgba(0,123,255,0.3);
            }
            
            .metric-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 5px solid #FF9900;
                margin: 0.75rem 0;
                transition: all 0.3s ease;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                border: 1px solid #e9ecef;
            }
            
            .metric-card:hover {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }
            
            .recommendation-box {
                background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 5px solid #007bff;
                margin: 1rem 0;
                box-shadow: 0 3px 15px rgba(0,123,255,0.1);
                border: 1px solid #b8daff;
            }
            
            .ai-insight {
                background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
                padding: 1.25rem;
                border-radius: 10px;
                border-left: 4px solid #007bff;
                margin: 1rem 0;
                font-style: italic;
                box-shadow: 0 2px 10px rgba(0,123,255,0.1);
                border: 1px solid #cce7ff;
            }
            
            .risk-high {
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                border-left: 5px solid #dc3545;
            }
            
            .risk-medium {
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                border-left: 5px solid #ffc107;
            }
            
            .risk-low {
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-left: 5px solid #28a745;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🗄️ Enterprise Database Migration Tool</h1>
            <p style="font-size: 1.1rem; margin-top: 0.5rem;">AI-Powered Sizing • Real-time Pricing • Risk Assessment • Migration Planning</p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">MS SQL • PostgreSQL • Oracle → AWS RDS • Aurora • EC2</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_navigation(self):
        """Render navigation bar"""
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            if st.button("⚙️ Configuration", key="nav_config"):
                st.session_state.active_tab = "configuration"
        with col2:
            if st.button("📊 Sizing Analysis", key="nav_sizing"):
                st.session_state.active_tab = "sizing"
        with col3:
            if st.button("💰 Cost Analysis", key="nav_cost"):
                st.session_state.active_tab = "cost"
        with col4:
            if st.button("⚠️ Risk Assessment", key="nav_risk"):
                st.session_state.active_tab = "risk"
        with col5:
            if st.button("📋 Migration Plan", key="nav_plan"):
                st.session_state.active_tab = "plan"
        with col6:
            if st.button("📈 Dashboard", key="nav_dashboard"):
                st.session_state.active_tab = "dashboard"
    
    def render_configuration_tab(self):
        """Render configuration tab"""
        st.markdown('<div class="section-header">⚙️ Database Migration Configuration</div>', unsafe_allow_html=True)
        
        # Project Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Project Information")
            project_name = st.text_input("Project Name", value="DB Migration 2025")
            source_database = st.selectbox("Source Database", 
                ["Microsoft SQL Server", "PostgreSQL", "Oracle", "MySQL", "Other"])
            target_platform = st.selectbox("Target Platform",
                ["Amazon RDS", "Amazon Aurora", "EC2 with Database", "Hybrid"])
            migration_type = st.selectbox("Migration Type",
                ["Homogeneous", "Heterogeneous"])
            environment = st.selectbox("Environment",
                ["Development", "QA", "SQA", "Production"])
        
        with col2:
            st.subheader("📊 Database Characteristics")
            database_size_gb = st.number_input("Database Size (GB)", min_value=1, max_value=100000, value=500)
            workload_type = st.selectbox("Workload Type",
                ["OLTP", "OLAP", "Mixed", "Analytics", "Reporting"])
            concurrent_connections = st.number_input("Peak Concurrent Connections", min_value=1, max_value=10000, value=200)
            transactions_per_second = st.number_input("Peak TPS", min_value=1, max_value=100000, value=2000)
            read_query_percentage = st.slider("Read Query Percentage", min_value=0, max_value=100, value=70)
        
        # Advanced Configuration
        st.subheader("🔧 Advanced Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Performance Requirements**")
            schema_complexity = st.selectbox("Schema Complexity", ["Simple", "Moderate", "Complex"])
            high_availability = st.checkbox("Multi-AZ Required", value=True)
            encryption_required = st.checkbox("Encryption Required", value=True)
            backup_retention_days = st.number_input("Backup Retention (Days)", min_value=1, max_value=365, value=30)
        
        with col2:
            st.markdown("**Growth & Scaling**")
            annual_growth_rate = st.slider("Annual Growth Rate (%)", min_value=0, max_value=100, value=20) / 100
            auto_scaling_enabled = st.checkbox("Auto Scaling", value=True)
            read_replica_scaling = st.checkbox("Auto Read Replica Scaling", value=True)
            
        with col3:
            st.markdown("**Migration Parameters**")
            downtime_tolerance = st.selectbox("Downtime Tolerance", ["Zero", "Low", "Medium", "High"])
            migration_method = st.selectbox("Preferred Migration Method", ["DMS", "Native Tools", "Hybrid"])
            data_sync_method = st.selectbox("Data Sync Method", ["Continuous", "Batch", "One-time"])
        
        # vROps Integration
        st.subheader("📊 vROps Integration (Optional)")
        enable_vrops = st.checkbox("Enable vROps Data Integration")
        
        if enable_vrops:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_peak = st.number_input("Peak CPU (%)", min_value=0, max_value=100, value=85)
                cpu_avg = st.number_input("Average CPU (%)", min_value=0, max_value=100, value=45)
            
            with col2:
                memory_peak = st.number_input("Peak Memory (GB)", min_value=0, max_value=1000, value=64)
                memory_avg = st.number_input("Average Memory (GB)", min_value=0, max_value=1000, value=32)
            
            with col3:
                iops_peak = st.number_input("Peak IOPS", min_value=0, max_value=100000, value=5000)
                iops_avg = st.number_input("Average IOPS", min_value=0, max_value=100000, value=2000)
            
            vrops_data = {
                "cpu_peak": cpu_peak,
                "cpu_avg": cpu_avg,
                "memory_peak": memory_peak,
                "memory_avg": memory_avg,
                "iops_peak": iops_peak,
                "iops_avg": iops_avg,
                "growth_rate": annual_growth_rate
            }
        else:
            vrops_data = None
        
        # Network Configuration
        st.subheader("🌐 Network Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            connectivity_type = st.selectbox("Connectivity Type",
                ["Direct Connect", "VPN", "Public Internet", "AWS PrivateLink"])
            target_region = st.selectbox("Target AWS Region",
                ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1"])
        
        with col2:
            vpc_endpoints = st.checkbox("Use VPC Endpoints", value=True)
            enhanced_monitoring = st.checkbox("Enhanced Monitoring", value=True)
        
        # Business Configuration
        st.subheader("💼 Business Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            business_criticality = st.selectbox("Business Criticality", ["Low", "Medium", "High", "Critical"])
            user_base_size = st.selectbox("User Base Size", ["Small", "Medium", "Large", "Enterprise"])
            compliance_frameworks = st.multiselect("Compliance Requirements",
                ["SOX", "GDPR", "HIPAA", "PCI-DSS", "SOC2", "ISO27001", "FedRAMP"])
        
        with col2:
            team_experience = st.selectbox("Team Experience Level", ["Expert", "Experienced", "Novice"])
            rollback_plan = st.selectbox("Rollback Plan", ["Comprehensive", "Basic", "None"])
            testing_strategy = st.selectbox("Testing Strategy", ["Comprehensive", "Adequate", "Minimal"])
        
        # AI Configuration
        st.subheader("🤖 AI Configuration")
        enable_claude_ai = st.checkbox("Enable Claude AI Analysis")
        
        if enable_claude_ai and ANTHROPIC_AVAILABLE:
            claude_api_key = st.text_input("Claude API Key", type="password")
            ai_model = st.selectbox("AI Model", [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514", 
                "claude-3-5-sonnet-20241022"
            ])
        else:
            claude_api_key = ""
            ai_model = "claude-sonnet-4-20250514"
        
        # Compile configuration
        config = {
            "project_name": project_name,
            "source_database": source_database,
            "target_platform": target_platform,
            "migration_type": migration_type,
            "environment": environment,
            "database_size_gb": database_size_gb,
            "workload_type": workload_type,
            "concurrent_connections": concurrent_connections,
            "transactions_per_second": transactions_per_second,
            "read_query_percentage": read_query_percentage,
            "schema_complexity": schema_complexity,
            "high_availability": high_availability,
            "encryption_required": encryption_required,
            "backup_retention_days": backup_retention_days,
            "annual_growth_rate": annual_growth_rate,
            "auto_scaling_enabled": auto_scaling_enabled,
            "read_replica_scaling": read_replica_scaling,
            "downtime_tolerance": downtime_tolerance,
            "migration_method": migration_method,
            "data_sync_method": data_sync_method,
            "connectivity_type": connectivity_type,
            "target_region": target_region,
            "vpc_endpoints": vpc_endpoints,
            "enhanced_monitoring": enhanced_monitoring,
            "business_criticality": business_criticality,
            "user_base_size": user_base_size,
            "compliance_frameworks": compliance_frameworks,
            "team_experience": team_experience,
            "rollback_plan": rollback_plan,
            "testing_strategy": testing_strategy,
            "enable_claude_ai": enable_claude_ai,
            "claude_api_key": claude_api_key,
            "ai_model": ai_model,
            "vrops_data": vrops_data
        }
        
        # Save configuration and run analysis
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Running comprehensive database migration analysis..."):
                # Perform sizing analysis
                sizing_config = self.sizing_engine.calculate_sizing_requirements(config, vrops_data)
                
                # Calculate costs
                cost_analysis = self.cost_calculator.calculate_total_migration_cost(
                    sizing_config, config, sizing_config['growth_projections']
                )
                
                # Assess risks
                risk_assessment = self.risk_assessor.assess_migration_risks(config)
                
                # Generate migration plan
                migration_plan = self.migration_planner.generate_migration_plan(config, risk_assessment)
                
                # Store analysis results
                st.session_state.current_analysis = {
                    "config": config,
                    "sizing": sizing_config,
                    "costs": cost_analysis,
                    "risks": risk_assessment,
                    "plan": migration_plan,
                    "timestamp": datetime.now()
                }
                
                st.success("✅ Analysis completed! Navigate to other tabs to view results.")
    
    def render_sizing_tab(self):
        """Render sizing analysis tab"""
        if not st.session_state.current_analysis:
            st.warning("⚠️ Please run analysis in Configuration tab first.")
            return
        
        analysis = st.session_state.current_analysis
        sizing_config = analysis['sizing']
        config = analysis['config']
        
        st.markdown('<div class="section-header">📊 AI-Powered Database Sizing Analysis</div>', unsafe_allow_html=True)
        
        # Sizing Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            writer_instance = sizing_config['writer_instance']['instance_type']
            st.metric("Writer Instance", writer_instance)
        
        with col2:
            num_readers = len(sizing_config['reader_instances'])
            st.metric("Read Replicas", str(num_readers))
        
        with col3:
            storage_gb = sizing_config['storage_config']['allocated_storage_gb']
            st.metric("Storage (GB)", f"{storage_gb:,}")
        
        with col4:
            storage_type = sizing_config['storage_config']['storage_type']
            st.metric("Storage Type", storage_type)
        
        # Detailed Sizing Configuration
        st.markdown('<div class="section-header">🔍 Detailed Configuration</div>', unsafe_allow_html=True)
        
        # Writer Instance Details
        st.subheader("✍️ Writer Instance Configuration")
        writer_specs = sizing_config['writer_instance']['specs']
        
        writer_data = pd.DataFrame({
            "Metric": ["Instance Type", "vCPUs", "Memory (GB)", "Network Performance", "Use Case"],
            "Value": [
                sizing_config['writer_instance']['instance_type'],
                writer_specs['vcpu'],
                writer_specs['memory'],
                writer_specs['network'],
                writer_specs['use_case']
            ],
            "Utilization": [
                "N/A",
                f"{sizing_config['writer_instance']['cpu_utilization']:.1f}%",
                f"{sizing_config['writer_instance']['memory_utilization']:.1f}%",
                "Optimized",
                "Matched"
            ]
        })
        
        st.dataframe(writer_data, use_container_width=True, hide_index=True)
        
        # Reader Instances Details
        if sizing_config['reader_instances']:
            st.subheader("📖 Read Replica Configuration")
            
            reader_data = []
            for i, reader in enumerate(sizing_config['reader_instances']):
                reader_data.append({
                    "Replica": f"Reader-{i+1}",
                    "Instance Type": reader['instance_type'],
                    "Role": reader['role'],
                    "Cross-AZ": "Yes" if reader['cross_az'] else "No"
                })
            
            reader_df = pd.DataFrame(reader_data)
            st.dataframe(reader_df, use_container_width=True, hide_index=True)
        
        # Storage Configuration
        st.subheader("💾 Storage Configuration")
        storage_config = sizing_config['storage_config']
        
        storage_data = pd.DataFrame({
            "Setting": [
                "Storage Type", "Allocated Storage (GB)", "Provisioned IOPS", 
                "Multi-AZ", "Encryption", "Backup Retention (Days)"
            ],
            "Value": [
                storage_config['storage_type'],
                f"{storage_config['allocated_storage_gb']:,}",
                f"{storage_config['provisioned_iops']:,}",
                "Yes" if storage_config['multi_az'] else "No",
                "Yes" if storage_config['encryption_enabled'] else "No",
                storage_config['backup_retention_days']
            ]
        })
        
        st.dataframe(storage_data, use_container_width=True, hide_index=True)
        
        # Sizing Rationale
        st.markdown('<div class="section-header">🧠 AI Sizing Rationale</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🤖 Sizing Analysis:</strong><br>
            {sizing_config['sizing_rationale']}
        </div>
        """, unsafe_allow_html=True)
        
        # Growth Projections
        st.markdown('<div class="section-header">📈 Growth Projections</div>', unsafe_allow_html=True)
        
        growth_data = []
        for year, projection in sizing_config['growth_projections'].items():
            growth_data.append({
                "Year": year.replace('_', ' ').title(),
                "Storage (GB)": f"{projection['storage_gb']:,}",
                "Growth Factor": f"{projection['growth_factor']:.2f}x",
                "Cost Increase": projection['estimated_cost_increase'],
                "Compute Recommendation": projection['compute_recommendation']
            })
        
        growth_df = pd.DataFrame(growth_data)
        st.dataframe(growth_df, use_container_width=True, hide_index=True)
        
        # Performance Optimization Chart
        st.subheader("📊 Performance Optimization Visualization")
        
        # Create performance comparison chart
        scenarios = ["Current On-Premises", "Recommended AWS", "Optimized AWS"]
        performance_values = [100, 120, 150]  # Relative performance indices
        
        fig = go.Figure(data=[
            go.Bar(x=scenarios, y=performance_values, 
                   marker_color=['lightcoral', 'lightblue', 'lightgreen'])
        ])
        
        fig.update_layout(
            title="Performance Comparison: On-Premises vs AWS",
            yaxis_title="Relative Performance Index",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_cost_tab(self):
        """Render cost analysis tab"""
        if not st.session_state.current_analysis:
            st.warning("⚠️ Please run analysis in Configuration tab first.")
            return
        
        analysis = st.session_state.current_analysis
        cost_analysis = analysis['costs']
        
        st.markdown('<div class="section-header">💰 Comprehensive Cost Analysis</div>', unsafe_allow_html=True)
        
        # Cost Summary
        summary = cost_analysis['summary']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Cost", f"${summary['total_monthly']:,.0f}")
        
        with col2:
            st.metric("Annual Cost", f"${summary['total_yearly']:,.0f}")
        
        with col3:
            st.metric("Migration Cost", f"${summary['migration_one_time']:,.0f}")
        
        # Detailed Cost Breakdown
        st.markdown('<div class="section-header">📊 Monthly Cost Breakdown</div>', unsafe_allow_html=True)
        
        operational_costs = cost_analysis['operational_costs']
        
        # Create cost breakdown chart
        labels = ['Compute', 'Storage', 'Backups', 'Network']
        values = [
            operational_costs['breakdown']['writer_instance'] + operational_costs['breakdown']['reader_instances'],
            operational_costs['breakdown']['storage'] + operational_costs['breakdown']['iops'],
            operational_costs['breakdown']['backups'],
            cost_analysis['network_costs']['monthly_total']
        ]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title="Monthly Cost Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed cost table
        cost_details = pd.DataFrame({
            "Cost Category": [
                "Writer Instance", "Reader Instances", "Storage", "IOPS", 
                "Backups", "Network Connectivity", "VPC Endpoints"
            ],
            "Monthly Cost": [
                f"${operational_costs['breakdown']['writer_instance']:,.0f}",
                f"${operational_costs['breakdown']['reader_instances']:,.0f}",
                f"${operational_costs['breakdown']['storage']:,.0f}",
                f"${operational_costs['breakdown']['iops']:,.0f}",
                f"${operational_costs['breakdown']['backups']:,.0f}",
                f"${cost_analysis['network_costs']['connectivity_monthly']:,.0f}",
                f"${cost_analysis['network_costs']['vpc_endpoints_monthly']:,.0f}"
            ],
            "Annual Cost": [
                f"${operational_costs['breakdown']['writer_instance'] * 12:,.0f}",
                f"${operational_costs['breakdown']['reader_instances'] * 12:,.0f}",
                f"${operational_costs['breakdown']['storage'] * 12:,.0f}",
                f"${operational_costs['breakdown']['iops'] * 12:,.0f}",
                f"${operational_costs['breakdown']['backups'] * 12:,.0f}",
                f"${cost_analysis['network_costs']['connectivity_monthly'] * 12:,.0f}",
                f"${cost_analysis['network_costs']['vpc_endpoints_monthly'] * 12:,.0f}"
            ]
        })
        
        st.dataframe(cost_details, use_container_width=True, hide_index=True)
        
        # Migration Costs
        st.markdown('<div class="section-header">🚀 One-time Migration Costs</div>', unsafe_allow_html=True)
        
        migration_costs = cost_analysis['migration_costs']
        
        migration_data = pd.DataFrame({
            "Cost Category": [
                "DMS Services", "Professional Services", "Testing & Validation", 
                "Training", "Data Transfer"
            ],
            "Cost": [
                f"${migration_costs['dms_costs']:,.0f}",
                f"${migration_costs['professional_services']:,.0f}",
                f"${migration_costs['testing_costs']:,.0f}",
                f"${migration_costs['training_costs']:,.0f}",
                f"${cost_analysis['network_costs']['data_transfer_one_time']:,.0f}"
            ]
        })
        
        st.dataframe(migration_data, use_container_width=True, hide_index=True)
        
        # Growth Cost Projections
        st.markdown('<div class="section-header">📈 Cost Growth Projections</div>', unsafe_allow_html=True)
        
        growth_costs = cost_analysis['growth_costs']
        
        years = []
        monthly_costs = []
        yearly_costs = []
        
        for year, data in growth_costs.items():
            years.append(year.replace('_', ' ').title())
            monthly_costs.append(data['monthly_cost'])
            yearly_costs.append(data['yearly_cost'])
        
        # Create growth projection chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=monthly_costs,
            mode='lines+markers',
            name='Monthly Cost',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=[c/12 for c in yearly_costs],
            mode='lines+markers',
            name='Annual Cost (Monthly Avg)',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Cost Growth Projection",
            xaxis_title="Timeline",
            yaxis_title="Cost (USD)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost Optimization Recommendations
        st.markdown('<div class="section-header">💡 Cost Optimization Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = [
            "Consider Reserved Instances for 20-40% cost savings on compute",
            "Use Aurora Serverless for variable workloads to optimize costs",
            "Implement automated backup lifecycle policies",
            "Monitor and right-size instances based on actual usage",
            "Use Read Replicas in different regions for disaster recovery cost optimization"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")
    
    def render_risk_tab(self):
        """Render risk assessment tab"""
        if not st.session_state.current_analysis:
            st.warning("⚠️ Please run analysis in Configuration tab first.")
            return
        
        analysis = st.session_state.current_analysis
        risk_assessment = analysis['risks']
        
        st.markdown('<div class="section-header">⚠️ Comprehensive Risk Assessment</div>', unsafe_allow_html=True)
        
        # Overall Risk Summary
        overall_risk = risk_assessment['overall_risk']
        risk_color = {
            "low": "success",
            "medium": "warning", 
            "high": "error",
            "critical": "error"
        }.get(overall_risk, "info")
        
        if risk_color == "success":
            st.success(f"✅ Overall Risk Level: {overall_risk.upper()}")
        elif risk_color == "warning":
            st.warning(f"⚠️ Overall Risk Level: {overall_risk.upper()}")
        else:
            st.error(f"🚨 Overall Risk Level: {overall_risk.upper()}")
        
        # Risk Matrix
        st.markdown('<div class="section-header">📊 Risk Matrix</div>', unsafe_allow_html=True)
        
        risk_matrix = risk_assessment['risk_matrix']
        
        for risk_item in risk_matrix:
            risk_level = risk_item['level']
            
            if risk_level in ['high', 'critical']:
                risk_class = 'risk-high'
            elif risk_level == 'medium':
                risk_class = 'risk-medium'
            else:
                risk_class = 'risk-low'
            
            st.markdown(f"""
            <div class="recommendation-box {risk_class}">
                <strong>Category:</strong> {risk_item['category'].title()}<br>
                <strong>Risk:</strong> {risk_item['risk'].replace('_', ' ').title()}<br>
                <strong>Level:</strong> {risk_item['level'].upper()}<br>
                <strong>Impact:</strong> {risk_item['impact']}
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Category Details
        st.markdown('<div class="section-header">🔍 Detailed Risk Analysis</div>', unsafe_allow_html=True)
        
        risk_scores = risk_assessment['risk_scores']
        
        # Technical Risks
        st.subheader("⚙️ Technical Risks")
        for risk_name, risk_data in risk_scores['technical'].items():
            level = risk_data['level']
            impact = risk_data['impact']
            
            level_color = {
                'low': '🟢',
                'medium': '🟡', 
                'high': '🔴',
                'critical': '🆘'
            }.get(level, '🟡')
            
            st.write(f"{level_color} **{risk_name.replace('_', ' ').title()}**: {level.upper()} - {impact}")
        
        # Business Risks
        st.subheader("💼 Business Risks")
        for risk_name, risk_data in risk_scores['business'].items():
            level = risk_data['level']
            impact = risk_data['impact']
            
            level_color = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🔴', 
                'critical': '🆘'
            }.get(level, '🟡')
            
            st.write(f"{level_color} **{risk_name.replace('_', ' ').title()}**: {level.upper()} - {impact}")
        
        # Operational Risks
        st.subheader("🔧 Operational Risks")
        for risk_name, risk_data in risk_scores['operational'].items():
            level = risk_data['level']
            impact = risk_data['impact']
            
            level_color = {
                'low': '🟢',
                'medium': '🟡',
                'high': '🔴',
                'critical': '🆘'
            }.get(level, '🟡')
            
            st.write(f"{level_color} **{risk_name.replace('_', ' ').title()}**: {level.upper()} - {impact}")
        
        # Mitigation Strategies
        st.markdown('<div class="section-header">🛡️ Risk Mitigation Strategies</div>', unsafe_allow_html=True)
        
        mitigation_strategies = risk_assessment['mitigation_strategies']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚙️ Technical Mitigation")
            for strategy in mitigation_strategies['technical']:
                st.write(f"• {strategy}")
        
        with col2:
            st.subheader("💼 Business Mitigation")
            for strategy in mitigation_strategies['business']:
                st.write(f"• {strategy}")
        
        with col3:
            st.subheader("🔧 Operational Mitigation")
            for strategy in mitigation_strategies['operational']:
                st.write(f"• {strategy}")
        
        # Risk Visualization
        st.markdown('<div class="section-header">📈 Risk Visualization</div>', unsafe_allow_html=True)
        
        # Create risk level distribution chart
        risk_levels = [item['level'] for item in risk_matrix]
        risk_counts = {level: risk_levels.count(level) for level in ['low', 'medium', 'high', 'critical']}
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                marker_color=['green', 'yellow', 'orange', 'red']
            )
        ])
        
        fig.update_layout(
            title="Risk Level Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Number of Risks",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_plan_tab(self):
        """Render migration plan tab"""
        if not st.session_state.current_analysis:
            st.warning("⚠️ Please run analysis in Configuration tab first.")
            return
        
        analysis = st.session_state.current_analysis
        migration_plan = analysis['plan']
        
        st.markdown('<div class="section-header">📋 Comprehensive Migration Plan</div>', unsafe_allow_html=True)
        
        # Timeline Overview
        timeline = migration_plan['timeline']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Duration", f"{timeline['total_duration_weeks']} weeks")
        
        with col2:
            st.metric("Total Duration", f"{timeline['total_duration_months']} months")
        
        with col3:
            st.metric("Number of Phases", "5")
        
        # Phase Timeline
        st.markdown('<div class="section-header">📅 Migration Timeline</div>', unsafe_allow_html=True)
        
        # Create timeline visualization
        phases = []
        start_weeks = []
        durations = []
        
        for phase_name, phase_data in timeline.items():
            if phase_name not in ['total_duration_weeks', 'total_duration_months']:
                phases.append(phase_name.title())
                start_weeks.append(phase_data['start_week'])
                durations.append(phase_data['duration_weeks'])
        
        fig = go.Figure()
        
        for i, (phase, start, duration) in enumerate(zip(phases, start_weeks, durations)):
            fig.add_trace(go.Bar(
                name=phase,
                x=[duration],
                y=[phase],
                orientation='h',
                base=[start-1],
                marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            ))
        
        fig.update_layout(
            title="Migration Phase Timeline",
            xaxis_title="Weeks",
            height=400,
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Phase Information
        st.markdown('<div class="section-header">📋 Detailed Phase Breakdown</div>', unsafe_allow_html=True)
        
        phase_details = migration_plan['phase_details']
        
        for phase_name, phase_info in timeline.items():
            if phase_name not in ['total_duration_weeks', 'total_duration_months']:
                
                st.subheader(f"📌 {phase_name.title()} Phase")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**Duration:** {phase_info['duration_weeks']} weeks")
                    st.write(f"**Timeline:** Week {phase_info['start_week']}-{phase_info['end_week']}")
                    
                    st.write("**Key Activities:**")
                    for activity in phase_info['activities']:
                        st.write(f"• {activity}")
                
                with col2:
                    if phase_name in phase_details:
                        details = phase_details[phase_name]
                        
                        st.write("**Objectives:**")
                        for objective in details['objectives']:
                            st.write(f"• {objective}")
                        
                        st.write("**Success Criteria:**")
                        for criteria in details['success_criteria']:
                            st.write(f"• {criteria}")
        
        # Resource Requirements
        st.markdown('<div class="section-header">👥 Resource Requirements</div>', unsafe_allow_html=True)
        
        resource_req = migration_plan['resource_requirements']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Team Composition")
            for role in resource_req['team_composition']:
                st.write(f"• {role}")
            
            st.subheader("⏱️ Effort Estimation")
            effort = resource_req['estimated_effort']
            st.write(f"**Total Effort:** {effort['total_person_weeks']} person-weeks")
            
            for role, weeks in effort['key_roles'].items():
                st.write(f"• {role}: {weeks} weeks")
        
        with col2:
            st.subheader("☁️ AWS Services Required")
            for service in resource_req['aws_services']:
                st.write(f"• {service}")
            
            st.subheader("🛠️ Tools & Software")
            for tool in resource_req['tools_and_software']:
                st.write(f"• {tool}")
        
        # Success Criteria
        st.markdown('<div class="section-header">🎯 Overall Success Criteria</div>', unsafe_allow_html=True)
        
        success_criteria = migration_plan['success_criteria']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚙️ Technical Success")
            for criteria in success_criteria['technical']:
                st.write(f"• {criteria}")
        
        with col2:
            st.subheader("💼 Business Success")
            for criteria in success_criteria['business']:
                st.write(f"• {criteria}")
        
        with col3:
            st.subheader("🔧 Operational Success")
            for criteria in success_criteria['operational']:
                st.write(f"• {criteria}")
        
        # Risk Mitigation Plan
        st.markdown('<div class="section-header">🛡️ Risk Mitigation Plan</div>', unsafe_allow_html=True)
        
        risk_mitigation = migration_plan['risk_mitigation']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚙️ Technical Risks")
            for mitigation in risk_mitigation['technical']:
                st.write(f"• {mitigation}")
        
        with col2:
            st.subheader("💼 Business Risks")
            for mitigation in risk_mitigation['business']:
                st.write(f"• {mitigation}")
        
        with col3:
            st.subheader("🔧 Operational Risks")
            for mitigation in risk_mitigation['operational']:
                st.write(f"• {mitigation}")
    
    def render_dashboard_tab(self):
        """Render executive dashboard"""
        if not st.session_state.current_analysis:
            st.warning("⚠️ Please run analysis in Configuration tab first.")
            return
        
        analysis = st.session_state.current_analysis
        
        st.markdown('<div class="section-header">📈 Executive Dashboard</div>', unsafe_allow_html=True)
        
        # High-level metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cost = analysis['costs']['summary']['total_yearly']
            st.metric("Annual Cost", f"${total_cost:,.0f}")
        
        with col2:
            migration_time = analysis['plan']['timeline']['total_duration_months']
            st.metric("Migration Timeline", f"{migration_time} months")
        
        with col3:
            overall_risk = analysis['risks']['overall_risk']
            st.metric("Risk Level", overall_risk.upper())
        
        with col4:
            writer_instance = analysis['sizing']['writer_instance']['instance_type']
            st.metric("Primary Instance", writer_instance)
        
        # Executive Summary
        st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
        
        config = analysis['config']
        
        summary_text = f"""
        **Project:** {config['project_name']} - {config['environment']} Environment Migration
        
        **Source:** {config['source_database']} ({config['database_size_gb']:,} GB)
        
        **Target:** {config['target_platform']} in {config['target_region']}
        
        **Migration Type:** {config['migration_type']} migration with {config['downtime_tolerance'].lower()} downtime tolerance
        
        **Recommended Architecture:**
        - Writer: {analysis['sizing']['writer_instance']['instance_type']}
        - Readers: {len(analysis['sizing']['reader_instances'])} read replicas
        - Storage: {analysis['sizing']['storage_config']['allocated_storage_gb']:,} GB {analysis['sizing']['storage_config']['storage_type']}
        - Multi-AZ: {'Yes' if analysis['sizing']['storage_config']['multi_az'] else 'No'}
        
        **Cost Summary:**
        - Monthly Operating Cost: ${analysis['costs']['summary']['total_monthly']:,.0f}
        - One-time Migration Cost: ${analysis['costs']['summary']['migration_one_time']:,.0f}
        - 3-Year Total: ${analysis['costs']['summary']['total_yearly'] * 3:,.0f}
        
        **Timeline:** {analysis['plan']['timeline']['total_duration_months']} months across 5 phases
        
        **Risk Assessment:** {analysis['risks']['overall_risk'].upper()} overall risk level
        """
        
        st.markdown(f"""
        <div class="recommendation-box">
            {summary_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Key Recommendations
        st.markdown('<div class="section-header">💡 Key Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = [
            f"Implement {config['target_platform']} with {analysis['sizing']['writer_instance']['instance_type']} for optimal performance",
            f"Use {len(analysis['sizing']['reader_instances'])} read replicas to handle {config['read_query_percentage']}% read workload",
            f"Plan for {analysis['plan']['timeline']['total_duration_months']}-month migration timeline",
            f"Address {analysis['risks']['overall_risk']} risk factors through comprehensive mitigation strategies",
            "Consider Reserved Instances for 20-40% cost savings on compute resources",
            "Implement monitoring and alerting from day one for operational excellence"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Cost Visualization
        st.markdown('<div class="section-header">💰 Cost Analysis Overview</div>', unsafe_allow_html=True)
        
        # Create cost comparison chart
        costs = analysis['costs']
        
        categories = ['Year 1', 'Year 2', 'Year 3']
        values = []
        
        base_yearly = costs['summary']['total_yearly']
        
        for year in [1, 2, 3]:
            if f'year_{year}' in costs['growth_costs']:
                values.append(costs['growth_costs'][f'year_{year}']['yearly_cost'])
            else:
                growth_factor = (1 + analysis['config']['annual_growth_rate']) ** year
                values.append(base_yearly * growth_factor)
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="3-Year Cost Projection",
            yaxis_title="Annual Cost (USD)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Action Items
        st.markdown('<div class="section-header">✅ Next Steps & Action Items</div>', unsafe_allow_html=True)
        
        st.write("**Immediate Actions (Next 2 weeks):**")
        st.write("• Review and approve migration plan and budget")
        st.write("• Assemble migration team and assign roles")
        st.write("• Begin AWS account setup and IAM configuration")
        
        st.write("**Short-term Actions (Month 1):**")
        st.write("• Complete current state assessment and documentation")
        st.write("• Set up AWS target environment")
        st.write("• Configure migration tools and connectivity")
        
        st.write("**Medium-term Actions (Months 2-3):**")
        st.write("• Execute comprehensive testing phase")
        st.write("• Validate performance and functionality")
        st.write("• Prepare for production migration")
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        self.render_navigation()
        
        # Render appropriate tab
        if st.session_state.active_tab == "configuration":
            self.render_configuration_tab()
        elif st.session_state.active_tab == "sizing":
            self.render_sizing_tab()
        elif st.session_state.active_tab == "cost":
            self.render_cost_tab()
        elif st.session_state.active_tab == "risk":
            self.render_risk_tab()
        elif st.session_state.active_tab == "plan":
            self.render_plan_tab()
        elif st.session_state.active_tab == "dashboard":
            self.render_dashboard_tab()
        
        # Footer
        st.markdown("---")
        st.markdown("**🗄️ Enterprise Database Migration Tool** - AI-Powered • Real-time Pricing • Comprehensive Analysis")

def main():
    """Main function"""
    try:
        platform = DatabaseMigrationPlatform()
        platform.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your configuration and try again.")

if __name__ == "__main__":
    main()