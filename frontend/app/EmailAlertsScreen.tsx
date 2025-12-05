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
  shadow: '#f9b9b9ff',
};

// Helper functions
const getRiskColor = (risk: string) => {
  switch (risk.toLowerCase()) {
    case 'high': return colors.alertRed;
    case 'medium': return colors.alertOrange;
    case 'low': return colors.alertGreen;
    default: return colors.textMedium;
  }
};

const getRiskBg = (risk: string) => {
  switch (risk.toLowerCase()) {
    case 'high': return '#FFF5F5';
    case 'medium': return '#FFF9F0';
    case 'low': return '#F0FFF4';
    default: return colors.white;
  }
};

// Elder care-specific mock email scams
const mockEmails = [
  {
    id: "1",
    title: "Medicare Benefits Update",
    sender: "medicare-benefits@gov-update.net",
    subject: "URGENT: Your Medicare Coverage Expires Today",
    preview: "Dear Beneficiary, Your Medicare Part B coverage will be terminated unless you verify your information immediately...",
    risk: "High",
    time: "2 min ago",
    resident: "Dorothy Johnson",
    room: "204",
    category: "Government Impersonation",
    aiConfidence: 98,
  },
  {
    id: "2",
    title: "Grandchild Emergency",
    sender: "emergency-help@quickwire.com",
    subject: "Grandma, I need help urgently!",
    preview: "Hi Grandma, it's me. I'm in trouble and need $2,000 wired immediately. Please don't tell mom and dad...",
    risk: "High",
    time: "15 min ago",
    resident: "Margaret Wilson",
    room: "118",
    category: "Grandparent Scam",
    aiConfidence: 96,
  },
  {
    id: "3",
    title: "Social Security Administration",
    sender: "ssa-alert@secure-ssa.org",
    subject: "Your Social Security Number Has Been Suspended",
    preview: "This is to inform you that your Social Security Number has been suspended due to suspicious activity...",
    risk: "High",
    time: "32 min ago",
    resident: "Robert Thompson",
    room: "305",
    category: "Government Impersonation",
    aiConfidence: 99,
  },
  {
    id: "4",
    title: "Pharmacy Prescription Alert",
    sender: "rxrefill@pharmacy-alerts.com",
    subject: "Action Required: Your Prescription is Ready",
    preview: "Your prescription refill is ready. To confirm delivery, please update your payment information...",
    risk: "Medium",
    time: "1 hour ago",
    resident: "Helen Martinez",
    room: "122",
    category: "Healthcare Scam",
    aiConfidence: 87,
  },
  {
    id: "5",
    title: "Publisher's Clearing House",
    sender: "winner@pch-prize.net",
    subject: "🎉 CONGRATULATIONS! You've Won $2.5 Million!",
    preview: "You have been selected as our grand prize winner! Pay the $499 processing fee via gift card...",
    risk: "High",
    time: "2 hours ago",
    resident: "Walter Davis",
    room: "201",
    category: "Lottery/Prize Scam",
    aiConfidence: 99,
  },
  {
    id: "6",
    title: "Apple Support",
    sender: "support@apple-id-verify.com",
    subject: "Your Apple ID Has Been Locked",
    preview: "We detected unauthorized access to your Apple account. Click here to verify your identity...",
    risk: "Medium",
    time: "3 hours ago",
    resident: "Betty Anderson",
    room: "156",
    category: "Tech Support Scam",
    aiConfidence: 91,
  },
];

export default function EmailAlertsScreen({ navigation }: any) {
  const [activeFilter, setActiveFilter] = useState('all');

  const highRiskCount = mockEmails.filter(e => e.risk === "High").length;
  const mediumRiskCount = mockEmails.filter(e => e.risk === "Medium").length;

  const filteredEmails = mockEmails.filter(email => {
    if (activeFilter === 'all') return true;
    return email.risk.toLowerCase() === activeFilter;
  });

  const stats = [
    { label: "High Risk", value: highRiskCount.toString(), color: colors.alertRed },
    { label: "Medium", value: mediumRiskCount.toString(), color: colors.alertOrange },
    { label: "Total", value: mockEmails.length.toString(), color: colors.accent },
  ];

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <TouchableOpacity onPress={() => navigation?.goBack()}>
            <Text style={styles.backButton}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Email Alerts</Text>
          <Text style={styles.headerSubtitle}>Flagged emails requiring review</Text>
        </View>
        <View style={styles.logoContainer}>
          <Text style={styles.logoIcon}>📧</Text>
        </View>
      </View>

      {/* Stats Card */}
      <View style={styles.statsCard}>
        <View style={styles.statsContent}>
          <View style={styles.statsIconContainer}>
            <Text style={styles.statsIcon}>🛡️</Text>
          </View>
          <View style={styles.statsTextContainer}>
            <Text style={styles.statsNumber}>{mockEmails.length}</Text>
            <Text style={styles.statsLabel}>Emails Flagged</Text>
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

      {/* Filter Pills */}
      <View style={styles.filterSection}>
        <Text style={styles.sectionTitle}>Filter by Risk</Text>
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
      </View>

      {/* Alerts List */}
      <View style={styles.alertsSection}>
        <Text style={styles.sectionTitle}>Flagged Emails</Text>
        <View style={styles.alertsList}>
          {filteredEmails.map((email) => (
            <TouchableOpacity
              key={email.id}
              style={[
                styles.alertCard,
                { 
                  borderLeftColor: getRiskColor(email.risk),
                  backgroundColor: getRiskBg(email.risk)
                }
              ]}
              onPress={() => navigation?.navigate("EmailDetails", { email })}
            >
              <View style={styles.alertCardHeader}>
                <View style={styles.alertCardTop}>
                  <View style={[styles.riskBadge, { backgroundColor: getRiskColor(email.risk) }]}>
                    <Text style={styles.riskBadgeText}>{email.risk.toUpperCase()}</Text>
                  </View>
                  <Text style={styles.alertTime}>{email.time}</Text>
                </View>
              </View>

              <View style={styles.alertCardBody}>
                <Text style={styles.alertTitle}>{email.title}</Text>
                <Text style={styles.alertSubject} numberOfLines={1}>{email.subject}</Text>
                <View style={styles.categoryTag}>
                  <Text style={styles.categoryText}>{email.category}</Text>
                </View>
              </View>

              <View style={styles.residentInfo}>
                <Text style={styles.residentName}>👤 {email.resident}</Text>
                <Text style={styles.residentRoom}>Room {email.room}</Text>
              </View>

              <View style={styles.alertCardFooter}>
                <View style={styles.confidenceRow}>
                  <Text style={styles.confidenceLabel}>AI Confidence: </Text>
                  <Text style={[styles.confidenceValue, { color: getRiskColor(email.risk) }]}>
                    {email.aiConfidence}%
                  </Text>
                </View>
                <Text style={styles.alertActionButton}>Review →</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Empty State */}
      {filteredEmails.length === 0 && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>✨</Text>
          <Text style={styles.emptyStateText}>No emails match this filter</Text>
        </View>
      )}

      <View style={{ height: 40 }} />
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
    alignItems: 'flex-start',
    marginBottom: 24,
    paddingVertical: 16,
  },
  headerLeft: {
    flexDirection: 'column',
    flex: 1,
  },
  backButton: {
    fontSize: 14,
    color: colors.accent,
    fontWeight: '600',
    marginBottom: 8,
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

  // Stats Card
  statsCard: {
    backgroundColor: colors.white,
    borderRadius: 20,
    padding: 24,
    marginBottom: 24,
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
    width: 64,
    height: 64,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accentLight,
    borderRadius: 16,
    marginRight: 20,
  },
  statsIcon: {
    fontSize: 36,
  },
  statsTextContainer: {
    flexDirection: 'column',
  },
  statsNumber: {
    fontSize: 36,
    fontWeight: '800',
    color: colors.textDark,
    lineHeight: 40,
  },
  statsLabel: {
    fontSize: 14,
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
    fontSize: 22,
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

  // Filter Section
  filterSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: 16,
  },
  filterContainer: {
    flexDirection: 'row',
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

  // Alerts Section
  alertsSection: {
    marginBottom: 24,
  },
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
    marginBottom: 12,
  },
  alertTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: 4,
  },
  alertSubject: {
    fontSize: 14,
    color: colors.textMedium,
    fontStyle: 'italic',
    marginBottom: 8,
  },
  categoryTag: {
    backgroundColor: colors.primaryLight,
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  categoryText: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.textDark,
  },
  residentInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: colors.background,
    padding: 12,
    borderRadius: 10,
    marginBottom: 12,
  },
  residentName: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.textDark,
  },
  residentRoom: {
    fontSize: 14,
    color: colors.textMedium,
  },
  alertCardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: colors.primaryLight,
    paddingTop: 12,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  confidenceLabel: {
    fontSize: 12,
    color: colors.textMedium,
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: '700',
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
