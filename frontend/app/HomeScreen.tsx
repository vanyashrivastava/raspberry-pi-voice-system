import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';

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
  shadow: '#000000',
};

// Helper function to determine risk color
const getRiskColor = (risk: string) => {
  switch (risk.toLowerCase()) {
    case 'high': return colors.alertRed;
    case 'medium': return colors.alertOrange;
    case 'low': return colors.alertGreen;
    default: return colors.textMedium;
  }
};

// Helper function to get risk background
const getRiskBg = (risk: string) => {
  switch (risk.toLowerCase()) {
    case 'high': return '#FFF5F5';
    case 'medium': return '#FFF9F0';
    case 'low': return '#F0FFF4';
    default: return colors.white;
  }
};

export default function HomeScreen({ navigation }: any) {
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
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.headerTitle}>{nursingHomeName}</Text>
          <Text style={styles.headerSubtitle}>Nursing Home</Text>
        </View>
        <View style={styles.logoContainer}>
          <Text style={styles.logoIcon}>🐷</Text>
        </View>
      </View>

      {/* Main Stats Card */}
      <View style={styles.mainStatsCard}>
        <View style={styles.statsContent}>
          <View style={styles.statsIconContainer}>
            <Text style={styles.statsIcon}>🛡️</Text>
          </View>
          <View style={styles.statsTextContainer}>
            <Text style={styles.statsNumber}>{residentsMonitored}</Text>
            <Text style={styles.statsLabel}>Residents Protected</Text>
          </View>
        </View>
        <View style={styles.statsSubInfo}>
          {stats.map((stat, index) => (
            <View key={index} style={styles.miniStat}>
              <Text style={[styles.miniStatValue, { color: stat.color }]}>{stat.value}</Text>
              <Text style={styles.miniStatLabel}>{stat.label}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Quick Actions Grid */}
      <View style={styles.quickActionsContainer}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.menuGrid}>
          {menuItems.map((item, index) => (
            <TouchableOpacity
              key={index}
              style={styles.menuButton}
              onPress={() => navigation?.navigate(item.screen)}
            >
              <Text style={styles.menuIcon}>{item.icon}</Text>
              <Text style={styles.menuLabel}>{item.label}</Text>
              <Text style={styles.menuArrow}>→</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Recent Alerts Section */}
      <View style={styles.alertsSection}>
        <View style={styles.alertsHeader}>
          <Text style={styles.sectionTitle}>Recent Alerts</Text>
          <TouchableOpacity style={styles.viewAllButton}>
            <Text style={styles.viewAllButtonText}>View All</Text>
          </TouchableOpacity>
        </View>

        {/* Filter Pills */}
        <View style={styles.filterContainer}>
          {['all', 'high', 'medium', 'low'].map((filter) => (
            <TouchableOpacity
              key={filter}
              style={[
                styles.filterPill,
                activeFilter === filter && styles.filterPillActive
              ]}
              onPress={() => setActiveFilter(filter)}
            >
              <Text style={[
                styles.filterPillText,
                activeFilter === filter && styles.filterPillTextActive
              ]}>
                {filter.charAt(0).toUpperCase() + filter.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Alerts List */}
        <View style={styles.alertsList}>
          {scamAlerts.map((alert) => (
            <TouchableOpacity
              key={alert.id}
              style={[
                styles.alertCard,
                { 
                  borderLeftColor: getRiskColor(alert.risk),
                  backgroundColor: getRiskBg(alert.risk)
                }
              ]}
              onPress={() => console.log('View alert details', alert.id)}
            >
              <View style={styles.alertCardHeader}>
                <View style={styles.alertCardTop}>
                  <View style={[
                    styles.riskBadge,
                    { backgroundColor: getRiskColor(alert.risk) }
                  ]}>
                    <Text style={styles.riskBadgeText}>{alert.risk.toUpperCase()}</Text>
                  </View>
                  <Text style={styles.alertTime}>{alert.time}</Text>
                </View>
              </View>
              
              <View style={styles.alertCardBody}>
                <Text style={styles.alertResidentName}>{alert.resident}</Text>
                <Text style={styles.alertType}>{alert.type}</Text>
              </View>

              <View style={styles.alertCardFooter}>
                <Text style={styles.alertActionButton}>Review Details →</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Empty state if no alerts */}
      {scamAlerts.length === 0 && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>🎉</Text>
          <Text style={styles.emptyStateText}>All clear! No active alerts.</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: 20,
    paddingTop: 50,
  },

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
    paddingVertical: 16,
  },
  headerLeft: {
    flexDirection: 'column',
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: '800',
    color: colors.textDark,
    letterSpacing: -0.5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: colors.textMedium,
    fontWeight: '500',
    marginTop: 2,
  },
  logoContainer: {
    width: 56,
    height: 56,
    borderRadius: 16,
    backgroundColor: colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  logoIcon: {
    fontSize: 28,
  },

  // Main Stats Card
  mainStatsCard: {
    backgroundColor: colors.white,
    borderRadius: 20,
    padding: 24,
    marginBottom: 32,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 4,
    borderWidth: 2,
    borderColor: colors.primaryLight,
  },
  statsContent: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: colors.primaryLight,
  },
  statsIconContainer: {
    width: 72,
    height: 72,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accentLight,
    borderRadius: 16,
    marginRight: 20,
  },
  statsIcon: {
    fontSize: 48,
  },
  statsTextContainer: {
    flexDirection: 'column',
  },
  statsNumber: {
    fontSize: 40,
    fontWeight: '800',
    color: colors.textDark,
    lineHeight: 44,
  },
  statsLabel: {
    fontSize: 16,
    color: colors.textMedium,
    fontWeight: '600',
  },
  statsSubInfo: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  miniStat: {
    alignItems: 'center',
    flex: 1,
  },
  miniStatValue: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 4,
  },
  miniStatLabel: {
    fontSize: 11,
    color: colors.textMedium,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },

  // Quick Actions
  quickActionsContainer: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: 16,
  },
  menuGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  menuButton: {
    width: '48%',
    backgroundColor: colors.white,
    borderRadius: 16,
    padding: 20,
    marginBottom: 12,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  menuIcon: {
    fontSize: 28,
    marginBottom: 8,
  },
  menuLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.textDark,
  },
  menuArrow: {
    position: 'absolute',
    bottom: 16,
    right: 16,
    fontSize: 18,
    color: colors.textMedium,
    opacity: 0.5,
  },

  // Alerts Section
  alertsSection: {
    marginBottom: 24,
  },
  alertsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  viewAllButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  viewAllButtonText: {
    color: colors.accent,
    fontSize: 14,
    fontWeight: '600',
  },

  // Filter Pills
  filterContainer: {
    flexDirection: 'row',
    marginBottom: 20,
    gap: 8,
  },
  filterPill: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: colors.primaryLight,
    backgroundColor: colors.white,
  },
  filterPillActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  filterPillText: {
    color: colors.textMedium,
    fontSize: 13,
    fontWeight: '600',
  },
  filterPillTextActive: {
    color: colors.white,
  },

  // Alert Cards
  alertsList: {
    gap: 12,
  },
  alertCard: {
    backgroundColor: colors.white,
    borderRadius: 16,
    padding: 20,
    borderLeftWidth: 6,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
    marginBottom: 12,
  },
  alertCardHeader: {
    marginBottom: 12,
  },
  alertCardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  riskBadge: {
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 12,
  },
  riskBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.white,
    letterSpacing: 0.5,
  },
  alertTime: {
    fontSize: 12,
    color: colors.textMedium,
    fontWeight: '500',
  },
  alertCardBody: {
    marginBottom: 16,
  },
  alertResidentName: {
    fontSize: 20,
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: 4,
  },
  alertType: {
    fontSize: 14,
    color: colors.textMedium,
    fontStyle: 'italic',
  },
  alertCardFooter: {
    borderTopWidth: 1,
    borderTopColor: colors.primaryLight,
    paddingTop: 12,
  },
  alertActionButton: {
    color: colors.accent,
    fontSize: 14,
    fontWeight: '600',
  },

  // Empty State
  emptyState: {
    alignItems: 'center',
    padding: 60,
    backgroundColor: colors.white,
    borderRadius: 20,
    marginTop: 20,
  },
  emptyStateIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyStateText: {
    fontSize: 16,
    color: colors.textMedium,
  },
});