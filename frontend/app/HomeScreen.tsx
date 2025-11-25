import React, { useState } from 'react';

// Professional color palette matching landing screen
const colors = {
  primary: '#FF9ECD',
  primaryLight: '#FFE5F1',
  accent: '#00D4AA',
  accentLight: '#E0FFF9',
  textDark: '#2D3436',
  textMedium: '#636E72',
  white: '#FFFFFF',
  background: '#FAFBFC',
  alertRed: '#FF6B6B',
  alertOrange: '#FFA94D',
  alertGreen: '#51CF66',
  shadow: 'rgba(0, 0, 0, 0.08)',
};

// Helper function to determine risk color
const getRiskColor = (risk) => {
  switch (risk.toLowerCase()) {
    case 'high': return colors.alertRed;
    case 'medium': return colors.alertOrange;
    case 'low': return colors.alertGreen;
    default: return colors.textMedium;
  }
};

// Helper function to get risk background
const getRiskBg = (risk) => {
  switch (risk.toLowerCase()) {
    case 'high': return '#FFF5F5';
    case 'medium': return '#FFF9F0';
    case 'low': return '#F0FFF4';
    default: return colors.white;
  }
};

export default function HomeScreen() {
  const [activeFilter, setActiveFilter] = useState('all');
  
  const nursingHomeName = "Silver Oaks";
  const residentsMonitored = 42;

  const scamAlerts = [
    { id: "1", resident: "Evelyn Carter", risk: "High", type: "Bank phishing text", time: "2 hours ago" },
    { id: "2", resident: "Howard Miles", risk: "Medium", type: "Medicare scam call", time: "5 hours ago" },
    { id: "3", resident: "Martha Lee", risk: "Low", type: "Junk email (not scam)", time: "Yesterday" },
  ];

  const menuItems = [
    { label: "Email Alerts", icon: '📧', screen: "EmailAlerts", color: colors.primary },
    { label: "Call Monitor", icon: '📞', screen: "CallMonitoring", color: '#8B5CF6' },
    { label: "Residents", icon: '👥', screen: "ResidentList", color: '#3B82F6' },
    { label: "Settings", icon: '⚙️', screen: "Settings", color: colors.textMedium },
  ];

  const stats = [
    { label: "Active Alerts", value: "2", color: colors.alertOrange },
    { label: "Resolved", value: "18", color: colors.alertGreen },
    { label: "This Week", value: "8", color: colors.accent },
  ];

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.headerTitle}>{nursingHomeName}</h1>
          <p style={styles.headerSubtitle}>Nursing Home</p>
        </div>
        <div style={styles.logoContainer}>
          <span style={styles.logoIcon}>🐷</span>
        </div>
      </div>

      {/* Main Stats Card */}
      <div style={styles.mainStatsCard}>
        <div style={styles.statsContent}>
          <div style={styles.statsIcon}>🛡️</div>
          <div style={styles.statsTextContainer}>
            <div style={styles.statsNumber}>{residentsMonitored}</div>
            <div style={styles.statsLabel}>Residents Protected</div>
          </div>
        </div>
        <div style={styles.statsSubInfo}>
          {stats.map((stat, index) => (
            <div key={index} style={styles.miniStat}>
              <div style={{...styles.miniStatValue, color: stat.color}}>{stat.value}</div>
              <div style={styles.miniStatLabel}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions Grid */}
      <div style={styles.quickActionsContainer}>
        <h2 style={styles.sectionTitle}>Quick Actions</h2>
        <div style={styles.menuGrid}>
          {menuItems.map((item, index) => (
            <button
              key={index}
              style={{...styles.menuButton, '--hover-color': item.color}}
              onClick={() => console.log(`Navigate to ${item.screen}`)}
            >
              <span style={styles.menuIcon}>{item.icon}</span>
              <span style={styles.menuLabel}>{item.label}</span>
              <span style={styles.menuArrow}>→</span>
            </button>
          ))}
        </div>
      </div>

      {/* Recent Alerts Section */}
      <div style={styles.alertsSection}>
        <div style={styles.alertsHeader}>
          <h2 style={styles.sectionTitle}>Recent Alerts</h2>
          <button style={styles.viewAllButton}>View All</button>
        </div>

        {/* Filter Pills */}
        <div style={styles.filterContainer}>
          {['all', 'high', 'medium', 'low'].map((filter) => (
            <button
              key={filter}
              style={{
                ...styles.filterPill,
                ...(activeFilter === filter ? styles.filterPillActive : {})
              }}
              onClick={() => setActiveFilter(filter)}
            >
              {filter.charAt(0).toUpperCase() + filter.slice(1)}
            </button>
          ))}
        </div>

        {/* Alerts List */}
        <div style={styles.alertsList}>
          {scamAlerts.map((alert) => (
            <div
              key={alert.id}
              style={{
                ...styles.alertCard,
                borderLeftColor: getRiskColor(alert.risk),
                backgroundColor: getRiskBg(alert.risk),
              }}
              onClick={() => console.log('View alert details', alert.id)}
            >
              <div style={styles.alertCardHeader}>
                <div style={styles.alertCardTop}>
                  <span style={{
                    ...styles.riskBadge,
                    backgroundColor: getRiskColor(alert.risk),
                  }}>
                    {alert.risk.toUpperCase()}
                  </span>
                  <span style={styles.alertTime}>{alert.time}</span>
                </div>
              </div>
              
              <div style={styles.alertCardBody}>
                <div style={styles.alertResidentName}>{alert.resident}</div>
                <div style={styles.alertType}>{alert.type}</div>
              </div>

              <div style={styles.alertCardFooter}>
                <button style={styles.alertActionButton}>
                  Review Details →
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Empty state if no alerts */}
      {scamAlerts.length === 0 && (
        <div style={styles.emptyState}>
          <span style={styles.emptyStateIcon}>🎉</span>
          <p style={styles.emptyStateText}>All clear! No active alerts.</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: colors.background,
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },

  // Header
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px',
    padding: '16px 0',
  },
  headerLeft: {
    display: 'flex',
    flexDirection: 'column',
  },
  headerTitle: {
    fontSize: '32px',
    fontWeight: '800',
    color: colors.textDark,
    margin: '0 0 4px 0',
    letterSpacing: '-0.5px',
  },
  headerSubtitle: {
    fontSize: '14px',
    color: colors.textMedium,
    margin: 0,
    fontWeight: '500',
  },
  logoContainer: {
    width: '56px',
    height: '56px',
    borderRadius: '16px',
    backgroundColor: colors.primaryLight,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: `0 2px 8px ${colors.shadow}`,
  },
  logoIcon: {
    fontSize: '28px',
  },

  // Main Stats Card
  mainStatsCard: {
    backgroundColor: colors.white,
    borderRadius: '20px',
    padding: '24px',
    marginBottom: '32px',
    boxShadow: `0 2px 12px ${colors.shadow}`,
    border: `2px solid ${colors.primaryLight}`,
  },
  statsContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '20px',
    marginBottom: '20px',
    paddingBottom: '20px',
    borderBottom: `1px solid ${colors.primaryLight}`,
  },
  statsIcon: {
    fontSize: '48px',
    width: '72px',
    height: '72px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accentLight,
    borderRadius: '16px',
  },
  statsTextContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  statsNumber: {
    fontSize: '40px',
    fontWeight: '800',
    color: colors.textDark,
    lineHeight: '1',
  },
  statsLabel: {
    fontSize: '16px',
    color: colors.textMedium,
    fontWeight: '600',
  },
  statsSubInfo: {
    display: 'flex',
    justifyContent: 'space-around',
    gap: '16px',
  },
  miniStat: {
    textAlign: 'center',
    flex: 1,
  },
  miniStatValue: {
    fontSize: '24px',
    fontWeight: '700',
    marginBottom: '4px',
  },
  miniStatLabel: {
    fontSize: '11px',
    color: colors.textMedium,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },

  // Quick Actions
  quickActionsContainer: {
    marginBottom: '32px',
  },
  sectionTitle: {
    fontSize: '20px',
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: '16px',
    margin: '0 0 16px 0',
  },
  menuGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '12px',
  },
  menuButton: {
    backgroundColor: colors.white,
    border: 'none',
    borderRadius: '16px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    boxShadow: `0 2px 8px ${colors.shadow}`,
    position: 'relative',
    overflow: 'hidden',
  },
  menuIcon: {
    fontSize: '28px',
    marginBottom: '4px',
  },
  menuLabel: {
    fontSize: '15px',
    fontWeight: '600',
    color: colors.textDark,
  },
  menuArrow: {
    position: 'absolute',
    bottom: '16px',
    right: '16px',
    fontSize: '18px',
    color: colors.textMedium,
    opacity: 0.5,
  },

  // Alerts Section
  alertsSection: {
    marginBottom: '24px',
  },
  alertsHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  },
  viewAllButton: {
    backgroundColor: 'transparent',
    border: 'none',
    color: colors.accent,
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    padding: '8px 12px',
    borderRadius: '8px',
    transition: 'background-color 0.2s',
  },

  // Filter Pills
  filterContainer: {
    display: 'flex',
    gap: '8px',
    marginBottom: '20px',
    overflowX: 'auto',
    paddingBottom: '4px',
  },
  filterPill: {
    padding: '8px 16px',
    borderRadius: '20px',
    border: `2px solid ${colors.primaryLight}`,
    backgroundColor: colors.white,
    color: colors.textMedium,
    fontSize: '13px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.2s',
    whiteSpace: 'nowrap',
  },
  filterPillActive: {
    backgroundColor: colors.primary,
    color: colors.white,
    borderColor: colors.primary,
  },

  // Alert Cards
  alertsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  alertCard: {
    backgroundColor: colors.white,
    borderRadius: '16px',
    padding: '20px',
    borderLeft: '6px solid',
    cursor: 'pointer',
    transition: 'all 0.2s',
    boxShadow: `0 2px 8px ${colors.shadow}`,
  },
  alertCardHeader: {
    marginBottom: '12px',
  },
  alertCardTop: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  riskBadge: {
    fontSize: '11px',
    fontWeight: '700',
    color: colors.white,
    padding: '4px 10px',
    borderRadius: '12px',
    letterSpacing: '0.5px',
  },
  alertTime: {
    fontSize: '12px',
    color: colors.textMedium,
    fontWeight: '500',
  },
  alertCardBody: {
    marginBottom: '16px',
  },
  alertResidentName: {
    fontSize: '20px',
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: '4px',
  },
  alertType: {
    fontSize: '14px',
    color: colors.textMedium,
    fontStyle: 'italic',
  },
  alertCardFooter: {
    borderTop: `1px solid ${colors.primaryLight}`,
    paddingTop: '12px',
  },
  alertActionButton: {
    backgroundColor: 'transparent',
    border: 'none',
    color: colors.accent,
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    padding: '0',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },

  // Empty State
  emptyState: {
    textAlign: 'center',
    padding: '60px 20px',
    backgroundColor: colors.white,
    borderRadius: '20px',
    marginTop: '20px',
  },
  emptyStateIcon: {
    fontSize: '64px',
    display: 'block',
    marginBottom: '16px',
  },
  emptyStateText: {
    fontSize: '16px',
    color: colors.textMedium,
    margin: 0,
  },
};