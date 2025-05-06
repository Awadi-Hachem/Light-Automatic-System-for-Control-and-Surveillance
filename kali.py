import streamlit as st
import time
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, Ether, IPv6, conf, BOOTP
from scapy.layers.inet6 import ICMPv6EchoRequest, ICMPv6EchoReply
from scapy.layers.l2 import arping
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import socket
import psutil

# Configuration globale de Scapy
conf.sniff_promisc = True  # Mode promiscuous
conf.iface = None  # Interface par défaut

# Page configuration
st.set_page_config(
    page_title="L.A.S.Ca.S ",
    page_icon="🛡️",
    layout="wide"
)

# App title
st.title("🛡️ Light Automatic System for Control and Surveillance")
st.write("""
This application analyzes network traffic in real-time, detects cyber threats and displays results.
""")

# Load training and test data
@st.cache_data
def load_data():
    try:
        train_data = pd.read_csv("C:\\Users\\MSI\\Desktop\\Nouveau dossier\\Train_data.csv")
        test_data = pd.read_csv("C:\\Users\\MSI\\Desktop\\Nouveau dossier\\Test_data.csv")

        if 'class' not in train_data.columns:
            st.error("'class' column missing in train_data. Check CSV file.")
            st.stop()

        return train_data, test_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

train_data, test_data = load_data()

# Enhanced categorical encoding with unseen label handling
def encode_categorical_columns(df, label_encoders=None, fit=True):
    if label_encoders is None:
        label_encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        if col != 'class':
            # Handle missing values and convert to string
            df[col] = df[col].fillna('unknown').astype(str)

            if fit:
                # Training mode - fit new encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
            else:
                # Prediction mode - use existing encoder
                if col in label_encoders:
                    # Handle unseen values
                    seen_classes = set(label_encoders[col].classes_)
                    current_values = set(df[col].unique())
                    unseen_values = current_values - seen_classes

                    if unseen_values:
                        # Map unseen values to 'unknown'
                        df[col] = df[col].apply(lambda x: x if x in seen_classes else 'unknown')

                    # Ensure 'unknown' is in classes
                    if 'unknown' not in seen_classes:
                        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')

                    try:
                        df[col] = label_encoders[col].transform(df[col])
                    except ValueError:
                        # Fallback if encoding fails
                        df[col] = 0
                else:
                    # New column not in training data
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le

    return df, label_encoders

# Encode training data
train_data, label_encoders = encode_categorical_columns(train_data)
# Encode test data with same encoders
test_data, _ = encode_categorical_columns(test_data, label_encoders, fit=False)

# Prepare features and target
X_train = train_data.drop(columns=['class'])
y_train = train_data['class']

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train models
@st.cache_resource
def train_models():
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # SVM
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Neural Network
    nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    nn_model.fit(X_train_scaled, y_train)

    # ELM Classifier
    class ELM(BaseEstimator, ClassifierMixin):
        def __init__(self, n_hidden=100):
            self.n_hidden = n_hidden

        def fit(self, X, y):
            X, y = check_X_y(X, y, multi_output=True)
            self.classes_ = unique_labels(y)
            self.X_ = X
            self.y_ = y
            self.input_weights = np.random.randn(X.shape[1], self.n_hidden)
            self.bias = np.random.randn(self.n_hidden)
            H = np.tanh(np.dot(X, self.input_weights) + self.bias)
            self.output_weights = np.dot(np.linalg.pinv(H), pd.get_dummies(y).values)
            return self

        def predict(self, X):
            check_is_fitted(self)
            X = check_array(X)
            hidden_layer = np.tanh(np.dot(X, self.input_weights) + self.bias)
            output_layer = np.dot(hidden_layer, self.output_weights)
            return self.classes_[np.argmax(output_layer, axis=1)]

    elm_model = ELM(n_hidden=100)
    elm_model.fit(X_train_scaled, y_train)

    return rf_model, svm_model, nn_model, elm_model, X_train.columns

rf_model, svm_model, nn_model, elm_model, feature_columns = train_models()

models = {
    'Random Forest': rf_model,
    'SVM': svm_model,
    'Neural Network': nn_model,
    'ELM': elm_model
}

class NetFlowAnalyzer:
    def __init__(self):
        self.flows = {}
        self.start_time = time.time()
        self.local_ips = self.get_local_ips()
        self.local_macs = self.get_local_macs()
        self.discovered_hosts = set()
        self.subnet = self.detect_subnet()  # Initialisation du subnet

    def detect_subnet(self):
        """Detect the local subnet using available interfaces"""
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET and addr.address != '127.0.0.1':
                    if addr.netmask:
                        # Simple subnet detection (first three octets)
                        ip_parts = addr.address.split('.')
                        return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
        return "172.20.10.0/24"  # Default fallback subnet

    def get_local_ips(self):
        """Get local IP addresses including IPv6"""
        local_ips = set()
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    local_ips.add(addr.address)
                elif addr.family == socket.AF_INET6:
                    local_ips.add(addr.address.split('%')[0])  # Remove interface suffix
        return local_ips

    def get_local_macs(self):
        """Get local MAC addresses"""
        local_macs = set()
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == psutil.AF_LINK:
                    local_macs.add(addr.address.lower())
        return local_macs

    def get_protocol_name(self, protocol_num):
        """Convert protocol number to name"""
        protocol_map = {
            1: 'icmp', 6: 'tcp', 17: 'udp', 2: 'igmp',
            47: 'gre', 50: 'esp', 51: 'ah', 58: 'icmpv6',
            89: 'ospf', 132: 'sctp'
        }
        return protocol_map.get(protocol_num, str(protocol_num))

    def get_service_name(self, port, protocol):
        """Convert port to known service name"""
        try:
            port = int(port)
            service_map = {
                'tcp': {
                    20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
                    25: 'smtp', 53: 'dns', 80: 'http', 110: 'pop3',
                    143: 'imap', 443: 'https', 445: 'microsoft_ds',
                    993: 'imaps', 995: 'pop3s', 3306: 'mysql', 3389: 'rdp'
                },
                'udp': {
                    53: 'dns', 67: 'dhcp', 68: 'dhcp', 69: 'tftp',
                    123: 'ntp', 161: 'snmp', 162: 'snmptrap',
                    500: 'isakmp', 514: 'syslog', 520: 'rip'
                }
            }
            return service_map.get(protocol, {}).get(port, f'{protocol}_{port}')
        except:
            return f'{protocol}_{port}'

    def discover_hosts(self, interface):
        """Active network scanning to discover hosts"""
        st.info("Discovering local hosts via ARP scan...")
        try:
            conf.verb = 0  # Disable Scapy logs
            ans, unans = arping(net=self.subnet, iface=interface, timeout=2)

            for sent, received in ans:
                ip = received.psrc
                mac = received.hwsrc
                if ip not in self.discovered_hosts:
                    self.discovered_hosts.add(ip)
                    self.add_discovered_host(ip, mac, "ARP Scan")

        except Exception as e:
            st.warning(f"ARP scan failed: {e}")

    def add_discovered_host(self, ip, mac, source):
        """Add discovered host to flows"""
        flow_key = (ip, None, None, None, 'discovered')

        if flow_key not in self.flows:
            self.flows[flow_key] = {
                'timestamp': time.time(),
                'src_ip': ip,
                'dst_ip': None,
                'src_mac': mac,
                'dst_mac': None,
                'protocol_type': 'host_discovery',
                'service': source,
                'flag': 'Discovered',
                'length': 0,
                'is_local': ip in self.local_ips,
                'start_time': time.time(),
                'packet_count': 1,
                'total_length': 0,
                'duration': 0,
                'ip_version': 4
            }

    def packet_handler(self, packet):
        st.session_state.packet_count += 1

        flow_info = {
            'timestamp': time.time(),
            'src_ip': None, 'dst_ip': None,
            'protocol': None, 'protocol_type': 'unknown',
            'src_port': None, 'dst_port': None,
            'service': 'other', 'flag': 'unknown',
            'length': len(packet), 'is_local': False,
            'src_mac': None, 'dst_mac': None,
            'ip_version': None
        }

        # Ethernet layer (always present in live capture)
        if Ether in packet:
            flow_info['src_mac'] = packet[Ether].src.lower()
            flow_info['dst_mac'] = packet[Ether].dst.lower()
            flow_info['is_local'] = (flow_info['src_mac'] in self.local_macs or
                                    flow_info['dst_mac'] in self.local_macs)

        # IPv4
        if IP in packet:
            flow_info.update(self.process_ip_packet(packet[IP]))
            flow_info['ip_version'] = 4

        # IPv6
        elif IPv6 in packet:
            flow_info.update(self.process_ipv6_packet(packet[IPv6]))
            flow_info['ip_version'] = 6

        # ARP
        elif ARP in packet:
            flow_info.update(self.process_arp_packet(packet[ARP]))

        # DHCP
        elif BOOTP in packet:
            flow_info.update(self.process_dhcp_packet(packet[BOOTP]))

        # Other L2 protocols
        else:
            flow_info['protocol_type'] = 'other_l2'

        # Update or create flow
        flow_key = self.create_flow_key(flow_info)
        self.update_flow(flow_key, flow_info)

        # Periodic dataframe update
        if st.session_state.packet_count % 10 == 0:
            self.update_dataframe()

    def process_ip_packet(self, ip_packet):
        info = {
            'src_ip': ip_packet.src,
            'dst_ip': ip_packet.dst,
            'protocol': ip_packet.proto,
            'protocol_type': self.get_protocol_name(ip_packet.proto),
            'is_local': ip_packet.src in self.local_ips or ip_packet.dst in self.local_ips
        }

        # TCP
        if TCP in ip_packet:
            info.update(self.process_tcp_packet(ip_packet[TCP]))

        # UDP
        elif UDP in ip_packet:
            info.update(self.process_udp_packet(ip_packet[UDP]))

        # ICMP
        elif ICMP in ip_packet:
            info.update(self.process_icmp_packet(ip_packet[ICMP]))

        return info

    def process_ipv6_packet(self, ipv6_packet):
        info = {
            'src_ip': ipv6_packet.src,
            'dst_ip': ipv6_packet.dst,
            'protocol': ipv6_packet.nh,
            'protocol_type': f"ipv6_{self.get_protocol_name(ipv6_packet.nh)}",
            'is_local': ipv6_packet.src in self.local_ips or ipv6_packet.dst in self.local_ips
        }

        # TCP
        if TCP in ipv6_packet:
            info.update(self.process_tcp_packet(ipv6_packet[TCP]))

        # UDP
        elif UDP in ipv6_packet:
            info.update(self.process_udp_packet(ipv6_packet[UDP]))

        # ICMPv6
        elif ICMPv6 in ipv6_packet:
            info.update(self.process_icmpv6_packet(ipv6_packet[ICMPv6]))

        return info

    def process_tcp_packet(self, tcp_packet):
        info = {
            'src_port': tcp_packet.sport,
            'dst_port': tcp_packet.dport,
            'service': self.get_service_name(tcp_packet.dport, 'tcp'),
            'flag': self.get_tcp_flag(tcp_packet.flags)
        }
        return info

    def process_udp_packet(self, udp_packet):
        return {
            'src_port': udp_packet.sport,
            'dst_port': udp_packet.dport,
            'service': self.get_service_name(udp_packet.dport, 'udp'),
            'flag': 'SF'
        }

    def process_icmp_packet(self, icmp_packet):
        return {
            'service': 'icmp',
            'flag': f'icmp_type_{icmp_packet.type}'
        }

    def process_icmpv6_packet(self, icmpv6_packet):
        return {
            'service': 'icmpv6',
            'flag': f'icmpv6_type_{icmpv6_packet.type}'
        }

    def process_arp_packet(self, arp_packet):
        return {
            'protocol_type': 'arp',
            'service': 'arp',
            'flag': 'ARP',
            'src_ip': arp_packet.psrc,
            'dst_ip': arp_packet.pdst
        }

    def process_dhcp_packet(self, dhcp_packet):
        return {
            'protocol_type': 'dhcp',
            'service': 'dhcp',
            'flag': 'DHCP',
            'src_ip': dhcp_packet.yiaddr,
            'dst_ip': dhcp_packet.ciaddr
        }

    def get_tcp_flag(self, flags):
        if flags & 0x01: return 'FIN'
        elif flags & 0x02: return 'SYN'
        elif flags & 0x04: return 'RST'
        elif flags & 0x08: return 'PSH'
        elif flags & 0x10: return 'ACK'
        elif flags & 0x20: return 'URG'
        elif flags & 0x11: return 'FIN-ACK'
        elif flags & 0x12: return 'SYN-ACK'
        return 'OTHER'

    def create_flow_key(self, flow_info):
        """Create a unique flow key based on connection parameters"""
        return (
            flow_info['src_ip'], flow_info['dst_ip'],
            flow_info['src_port'], flow_info['dst_port'],
            flow_info['protocol_type'],
            flow_info['ip_version'] if 'ip_version' in flow_info else 4
        )

    def update_flow(self, flow_key, flow_info):
        if flow_key in self.flows:
            existing = self.flows[flow_key]
            existing['duration'] = flow_info['timestamp'] - existing['start_time']
            existing['packet_count'] += 1
            existing['total_length'] += flow_info['length']

            # Update flags to show most recent
            existing['flag'] = flow_info['flag']
        else:
            flow_info['start_time'] = flow_info['timestamp']
            flow_info['packet_count'] = 1
            flow_info['total_length'] = flow_info['length']
            flow_info['duration'] = 0
            self.flows[flow_key] = flow_info

    def update_dataframe(self):
        if not self.flows:
            return

        temp_df = pd.DataFrame(list(self.flows.values()))

        # Calculate flow metrics
        temp_df['packets_per_sec'] = temp_df['packet_count'] / (temp_df['duration'] + 0.001)
        temp_df['bytes_per_sec'] = temp_df['total_length'] / (temp_df['duration'] + 0.001)
        temp_df['bytes_per_packet'] = temp_df['total_length'] / temp_df['packet_count']

        st.session_state.df = temp_df.copy()

    def start_capture(self, duration=60, interface=None):
        st.info(f"Starting capture for {duration} seconds on interface {interface}...")
        st.session_state.packet_count = 0
        self.start_time = time.time()
        self.flows = {}  # Reset flows for new capture
        self.discovered_hosts = set()

        # Active host discovery before capture
        self.discover_hosts(interface)

        # Use a broader capture filter
        capture_filter = "ip or ip6 or arp or icmp or udp or tcp"

        try:
            sniff(prn=self.packet_handler,
                  timeout=duration,
                  iface=interface,
                  store=False,
                  filter=capture_filter,
                  promisc=True)  # Force promiscuous mode

            self.update_dataframe()
            st.success(f"Capture completed. {st.session_state.packet_count} packets analyzed.")
            self.analyze_traffic()
        except Exception as e:
            st.error(f"Capture error: {e}")
            st.error("Try running with administrator privileges.")

    def analyze_traffic(self):
        if st.session_state.df.empty:
            st.warning("No traffic data to analyze.")
            return

        # Display statistics
        st.subheader("📊 Traffic Statistics")
        cols = st.columns(4)
        cols[0].metric("Total Packets", st.session_state.packet_count)
        cols[1].metric("Unique Flows", len(st.session_state.df))
        cols[2].metric("Capture Duration", f"{time.time() - self.start_time:.2f} sec")

        # Calculate traffic volume
        total_bytes = st.session_state.df['total_length'].sum()
        cols[3].metric("Total Traffic", f"{total_bytes/1024:.2f} KB")

        # Display all discovered hosts
        st.subheader("🌐 All Discovered Hosts")
        hosts = set()
        for flow in self.flows.values():
            if flow['src_ip'] and flow['src_ip'] not in hosts:
                hosts.add(flow['src_ip'])
            if flow['dst_ip'] and flow['dst_ip'] not in hosts:
                hosts.add(flow['dst_ip'])

        # Add local IPs
        hosts.update(self.local_ips)

        if hosts:
            st.write(f"**Total hosts detected:** {len(hosts)}")
            cols = st.columns(4)
            for i, host in enumerate(sorted(hosts)):
                cols[i%4].code(host)



        # Raw data
        st.subheader("📊 Captured NetFlow Data")
        st.dataframe(st.session_state.df)

        # Visualizations
        self.show_protocol_distribution()
        self.show_service_distribution()
        self.show_traffic_volume()

        if len(st.session_state.df) > 0:
            self.detect_anomalies()

    def show_protocol_distribution(self):
        if 'protocol_type' in st.session_state.df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            protocol_counts = st.session_state.df['protocol_type'].value_counts()
            protocol_counts.plot(kind='bar', ax=ax)
            ax.set_title("Protocol Distribution")
            ax.set_ylabel("Number of Flows")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

    def show_service_distribution(self):
        if 'service' in st.session_state.df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            top_services = st.session_state.df['service'].value_counts().head(15)
            top_services.plot(kind='bar', ax=ax)
            ax.set_title("Top 15 Services")
            ax.set_ylabel("Number of Flows")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

    def show_traffic_volume(self):
        if not st.session_state.df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Group by protocol and sum traffic
            traffic_by_proto = st.session_state.df.groupby('protocol_type')['total_length'].sum().sort_values(ascending=False)

            # Convert to KB
            traffic_by_proto_kb = traffic_by_proto / 1024

            traffic_by_proto_kb.plot(kind='bar', ax=ax)
            ax.set_title("Traffic Volume by Protocol (KB)")
            ax.set_ylabel("Traffic Volume (KB)")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

    def detect_anomalies(self):
        st.subheader("🔮 Real-time Threat Detection")
        df_analysis = st.session_state.df.copy()

        # Clean data
        cols_to_drop = ['timestamp', 'src_mac', 'dst_mac', 'start_time', 'is_local', 'ip_version']
        df_analysis = df_analysis.drop(columns=[col for col in cols_to_drop if col in df_analysis.columns])

        # Encode categorical columns
        for col in ['protocol_type', 'service', 'flag']:
            if col in df_analysis.columns:
                df_analysis[col] = df_analysis[col].fillna('unknown').astype(str)
                if col in label_encoders:
                    # Handle unseen values
                    df_analysis[col] = df_analysis[col].apply(
                        lambda x: x if x in label_encoders[col].classes_ else 'unknown')
                    try:
                        df_analysis[col] = label_encoders[col].transform(df_analysis[col])
                    except ValueError:
                        df_analysis[col] = 0
                else:
                    le = LabelEncoder()
                    df_analysis[col] = le.fit_transform(df_analysis[col])
                    label_encoders[col] = le

        # Feature mapping
        feature_mapping = {
            'duration': 'duration',
            'protocol_type': 'protocol_type',
            'service': 'service',
            'flag': 'flag',
            'length': 'src_bytes',
            'total_length': 'dst_bytes',
            'packet_count': 'count',
            'packets_per_sec': 'same_srv_rate',
            'bytes_per_sec': 'dst_host_same_src_port_rate',
            'bytes_per_packet': 'dst_host_srv_count'
        }

        # Prepare final dataframe
        final_df = pd.DataFrame(0, index=df_analysis.index, columns=feature_columns)
        for src, target in feature_mapping.items():
            if src in df_analysis.columns and target in final_df.columns:
                final_df[target] = df_analysis[src]

        try:
            X = scaler.transform(final_df)
            self.make_predictions(X)
        except Exception as e:
            st.error(f"Anomaly detection error: {e}")

    def make_predictions(self, X):
        for name, model in models.items():
            try:
                predictions = model.predict(X)
                st.session_state.df[f'Prediction_{name}'] = [
                    '🚨 Threat detected' if p == 'anomaly' else '✅ Normal traffic'
                    for p in predictions
                ]

                st.write(f"**Detection results with {name}:**")
                st.dataframe(st.session_state.df[['src_ip', 'dst_ip', 'protocol_type', 'service', f'Prediction_{name}']])

                fig, ax = plt.subplots()
                sns.countplot(x=f'Prediction_{name}', data=st.session_state.df, ax=ax)
                ax.set_title(f"Prediction Distribution ({name})")
                st.pyplot(fig)

                if '🚨 Threat detected' in st.session_state.df[f'Prediction_{name}'].values:
                    st.warning(f"**Suspicious flows detected by {name}:**")
                    suspicious = st.session_state.df[st.session_state.df[f'Prediction_{name}'] == '🚨 Threat detected']
                    st.dataframe(suspicious[['src_ip', 'dst_ip', 'protocol_type', 'service', 'packet_count', 'total_length']])
            except Exception as e:
                st.error(f"Prediction error with {name}: {e}")

def get_network_interfaces():
    """Get available network interfaces with their IP addresses"""
    interfaces = []
    for interface, addrs in psutil.net_if_addrs().items():
        ips = []
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ips.append(addr.address)
            elif addr.family == socket.AF_INET6:
                ips.append(addr.address.split('%')[0])

        if ips:
            interfaces.append(f"{interface} ({', '.join(ips)})")
        else:
            interfaces.append(interface)
    return interfaces

def main():
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'packet_count' not in st.session_state:
        st.session_state.packet_count = 0

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Get interfaces with their IP addresses
    interface_list = get_network_interfaces()
    default_iface = next((iface for iface in interface_list if 'eth0' in iface or 'en0' in iface),
                         interface_list[0] if interface_list else 'eth0')

    # Extract just the interface name for Scapy
    selected_iface = st.sidebar.selectbox(
        "Network Interface",
        options=interface_list,
        index=interface_list.index(default_iface) if default_iface in interface_list else 0
    )
    interface_name = selected_iface.split(' ')[0]  # Get just the interface name

    duration = st.sidebar.slider("Capture Duration (seconds)", 10, 600, 60)
    promisc = st.sidebar.checkbox("Promiscuous Mode", value=True)

    analyzer = NetFlowAnalyzer()

    if st.sidebar.button("Start Capture"):
        conf.sniff_promisc = promisc
        analyzer.start_capture(duration, interface_name)

    # Continuous monitoring option
    continuous = st.sidebar.checkbox("Continuous Monitoring", False)
    if continuous:
        placeholder = st.empty()
        stop_button = st.sidebar.button("Stop Monitoring")

        while continuous and not stop_button:
            analyzer.start_capture(10, interface_name)  # Capture in 10-second intervals
            time.sleep(1)
            placeholder.empty()  # Clear previous output
            if stop_button:
                break

    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    cols = st.sidebar.columns([1, 4])
    cols[0].image("https://cdn-icons-png.flaticon.com/512/3135/3135823.png", width=60)
    cols[1].markdown("""
    **Author:** E. O. Hachem Aouadi
    **Advisors:**
     - Col. Dr. Mohamed Hechmi Jeridi
     - Col. Radhwen Hedi
     - lt. Ben Abdallah Asser
    """)

if __name__ == "__main__":
    main()
