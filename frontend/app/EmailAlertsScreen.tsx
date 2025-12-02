import React, { useState } from "react";
import { View, Text, StyleSheet, FlatList, TouchableOpacity, ScrollView, Animated } from "react-native";
import { colors } from "../theme/colors";

// Helper function to dynamically determine risk color
const getRiskColor = (risk) => {
  switch (risk.toLowerCase()) {
    case 'high':
      return colors.alertRed;
    case 'medium':
      return colors.alertOrange;
    case 'low':
      return colors.alertGreen;
    default:
      return colors.textSecondary;
  }
};

// Elder care-specific mock email scams
const mockEmails = [
  {
    id: "1",
    title: "Medicare Benefits Update",
    sender: "medicare-benefits@gov-update.net",
    senderReal: "Unknown (Spoofed)",
    subject: "URGENT: Your Medicare Coverage Expires Today",
    preview: "Dear Beneficiary, Your Medicare Part B coverage will be terminated unless you verify your information immediately. Click here to confirm your Social Security number and avoid losing your benefits...",
    risk: "High",
    timestamp: "2 min ago",
    resident: "Dorothy Johnson",
    room: "Room 204",
    category: "Government Impersonation",
    flags: ["Urgency tactics", "Requests SSN", "Spoofed sender", "Generic greeting"],
    aiConfidence: 98,
  },
  {
    id: "2",
    title: "Grandchild Emergency",
    sender: "emergency-help@quickwire.com",
    senderReal: "Unknown",
    subject: "Grandma, I need help urgently!",
    preview: "Hi Grandma, it's me. I'm in trouble and need $2,000 wired immediately. I got arrested and need bail money. Please don't tell mom and dad, I'm so embarrassed...",
    risk: "High",
    timestamp: "15 min ago",
    resident: "Margaret Wilson",
    room: "Room 118",
    category: "Grandparent Scam",
    flags: ["Emotional manipulation", "Requests wire transfer", "Secrecy request", "Vague identity"],
    aiConfidence: 96,
  },
  {
    id: "3",
    title: "Social Security Administration",
    sender: "ssa-alert@secure-ssa.org",
    senderReal: "Unknown (Spoofed)",
    subject: "Your Social Security Number Has Been Suspended",
    preview: "This is to inform you that your Social Security Number has been suspended due to suspicious activity. To reactivate your benefits, please call our secure line and verify your identity...",
    risk: "High",
    timestamp: "32 min ago",
    resident: "Robert Thompson",
    room: "Room 305",
    category: "Government Impersonation",
    flags: ["SSN suspension threat", "Phone callback request", "Fear tactics", "Spoofed government"],
    aiConfidence: 99,
  },
  {
    id: "4",
    title: "Pharmacy Prescription Alert",
    sender: "rxrefill@pharmacy-alerts.com",
    senderReal: "Unknown",
    subject: "Action Required: Your Prescription is Ready",
    preview: "Your prescription refill is ready for pickup. To confirm delivery to your home, please update your payment information and verify your Medicare ID number by clicking the link below...",
    risk: "Medium",
    timestamp: "1 hour ago",
    resident: "Helen Martinez",
    room: "Room 122",
    category: "Healthcare Scam",
    flags: ["Payment request", "Medicare ID request", "Suspicious link"],
    aiConfidence: 87,
  },
  {
    id: "5",
    title: "Publisher's Clearing House",
    sender: "winner@pch-prize.net",
    senderReal: "Unknown",
    subject: "🎉 CONGRATULATIONS! You've Won $2.5 Million!",
    preview: "You have been selected as our grand prize winner! To claim your $2,500,000 prize, please pay the $499 processing fee via gift card. This offer expires in 24 hours...",
    risk: "High",
    timestamp: "2 hours ago",
    resident: "Walter Davis",
    room: "Room 201",
    category: "Lottery/Prize Scam",
    flags: ["Unsolicited prize", "Upfront fee request", "Gift card payment", "Time pressure"],
    aiConfidence: 99,
  },
  {
    id: "6",
    title: "Apple Support",
    sender: "support@apple-id-verify.com",
    senderReal: "Unknown (Spoofed)",
    subject: "Your Apple ID Has Been Locked",
    preview: "We detected unauthorized access to your Apple account. Your account has been temporarily locked. Click here to verify your identity and restore access to your device...",
    risk: "Medium",
    timestamp: "3 hours ago",
    resident: "Betty Anderson",
    room: "Room 156",
    category: "Tech Support Scam",
    flags: ["Account lock threat", "Phishing link", "Spoofed tech company"],
    aiConfidence: 91,
  },
  {
    id: "7",
    title: "IRS Tax Notice",
    sender: "irs-refund@tax-gov.net",
    senderReal: "Unknown (Spoofed)",
    subject: "Unclaimed Tax Refund - $3,247.00",
    preview: "Our records indicate you have an unclaimed tax refund of $3,247.00. To process your refund, please verify your bank account information within 48 hours or your refund will be forfeited...",
    risk: "High",
    timestamp: "4 hours ago",
    resident: "George White",
    room: "Room 189",
    category: "Government Impersonation",
    flags: ["IRS impersonation", "Bank info request", "Time pressure", "Unsolicited refund"],
    aiConfidence: 97,
  },
  {
    id: "8",
    title: "Amazon Order Confirmation",
    sender: "orders@amazon-delivery.net",
    senderReal: "Unknown (Spoofed)",
    subject: "Your Order #789-4521 for $1,299.99 Has Shipped",
    preview: "Thank you for your purchase! Your order for MacBook Pro ($1,299.99) will arrive in 2 days. If you did not place this order, click here immediately to cancel and secure your account...",
    risk: "Medium",
    timestamp: "5 hours ago",
    resident: "Patricia Brown",
    room: "Room 167",
    category: "Fake Order Scam",
    flags: ["Fake order notification", "Panic inducement", "Phishing link"],
    aiConfidence: 89,
  },
];

export default function EmailAlertsScreen({ navigation }) {
  const [filter, setFilter] = useState("all");
  const [sortBy, setSortBy] = useState("time");

  const highRiskCount = mockEmails.filter(e => e.risk === "High").length;
  const mediumRiskCount = mockEmails.filter(e => e.risk === "Medium").length;
  const totalAlerts = mockEmails.length;

  const filteredEmails = mockEmails
    .filter(email => {
      if (filter === "all") return true;
      return email.risk.toLowerCase() === filter;
    })
    .sort((a, b) => {
      if (sortBy === "risk") {
        const riskOrder = { "High": 0, "Medium": 1, "Low": 2 };
        return riskOrder[a.risk] - riskOrder[b.risk];
      }
      return 0; // Default time sort (already sorted in mock data)
    });

  const filterOptions = [
    { key: "all", label: "All", count: totalAlerts },
    { key: "high", label: "High Risk", count: highRiskCount },
    { key: "medium", label: "Medium", count: mediumRiskCount },
  ];

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* HEADER */}
      <View style={styles.headerContainer}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.logoIcon}>🛡️</Text>
      </View>
      <Text style={styles.header}>Email Alerts</Text>
      <Text style={styles.subheader}>Flagged emails requiring staff attention</Text>

      {/* STATS CARDS ROW */}
      <View style={styles.statsRow}>
        <View style={[styles.statsCard, { backgroundColor: colors.alertRed + '20' }]}>
          <Text style={[styles.statsNumber, { color: colors.alertRed }]}>{highRiskCount}</Text>
          <Text style={styles.statsLabel}>🚨 High Risk</Text>
        </View>
        <View style={[styles.statsCard, { backgroundColor: colors.alertOrange + '20' }]}>
          <Text style={[styles.statsNumber, { color: colors.alertOrange }]}>{mediumRiskCount}</Text>
          <Text style={styles.statsLabel}>⚠️ Medium</Text>
        </View>
        <View style={[styles.statsCard, { backgroundColor: colors.lightGreen }]}>
          <Text style={[styles.statsNumber, { color: colors.textDark }]}>{totalAlerts}</Text>
          <Text style={styles.statsLabel}>📧 Total</Text>
        </View>
      </View>

      {/* FILTER PILLS */}
      <View style={styles.filterContainer}>
        <Text style={styles.filterLabel}>Filter by:</Text>
        <View style={styles.filterPills}>
          {filterOptions.map((option) => (
            <TouchableOpacity
              key={option.key}
              style={[
                styles.filterPill,
                filter === option.key && styles.filterPillActive
              ]}
              onPress={() => setFilter(option.key)}
            >
              <Text style={[
                styles.filterPillText,
                filter === option.key && styles.filterPillTextActive
              ]}>
                {option.label} ({option.count})
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* SORT TOGGLE */}
      <View style={styles.sortContainer}>
        <TouchableOpacity
          style={[styles.sortButton, sortBy === "time" && styles.sortButtonActive]}
          onPress={() => setSortBy("time")}
        >
          <Text style={[styles.sortButtonText, sortBy === "time" && styles.sortButtonTextActive]}>
            🕐 Most Recent
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.sortButton, sortBy === "risk" && styles.sortButtonActive]}
          onPress={() => setSortBy("risk")}
        >
          <Text style={[styles.sortButtonText, sortBy === "risk" && styles.sortButtonTextActive]}>
            🎯 By Risk Level
          </Text>
        </TouchableOpacity>
      </View>

      {/* ALERTS LIST */}
      <Text style={styles.sectionTitle}>⚠️ Flagged Emails</Text>
      {filteredEmails.map((item) => {
        const riskColor = getRiskColor(item.risk);
        return (
          <TouchableOpacity
            key={item.id}
            style={styles.alertCard(riskColor)}
            onPress={() => navigation.navigate("EmailDetails", { email: item })}
            activeOpacity={0.7}
          >
            {/* Card Header */}
            <View style={styles.alertHeader}>
              <View style={styles.riskBadge(riskColor)}>
                <Text style={styles.riskBadgeText}>{item.risk.toUpperCase()}</Text>
              </View>
              <Text style={styles.timestamp}>{item.timestamp}</Text>
            </View>

            {/* Category Tag */}
            <View style={styles.categoryTag}>
              <Text style={styles.categoryText}>{item.category}</Text>
            </View>

            {/* Email Title & Subject */}
            <Text style={styles.alertTitle}>{item.title}</Text>
            <Text style={styles.alertSubject}>{item.subject}</Text>

            {/* Preview */}
            <Text style={styles.alertPreview} numberOfLines={2}>
              {item.preview}
            </Text>

            {/* Resident Info */}
            <View style={styles.residentInfo}>
              <Text style={styles.residentName}>👤 {item.resident}</Text>
              <Text style={styles.residentRoom}>📍 {item.room}</Text>
            </View>

            {/* AI Confidence */}
            <View style={styles.confidenceBar}>
              <Text style={styles.confidenceLabel}>AI Confidence:</Text>
              <View style={styles.confidenceTrack}>
                <View style={[styles.confidenceFill, { width: `${item.aiConfidence}%`, backgroundColor: riskColor }]} />
              </View>
              <Text style={[styles.confidenceValue, { color: riskColor }]}>{item.aiConfidence}%</Text>
            </View>

            {/* Tap Indicator */}
            <Text style={styles.tapHint}>Tap to review →</Text>
          </TouchableOpacity>
        );
      })}

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingTop: 50,
    paddingHorizontal: 20,
  },
  headerContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  backButton: {
    fontSize: 16,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  logoIcon: {
    fontSize: 28,
  },
  header: {
    fontSize: 28,
    fontWeight: "800",
    color: colors.textDark,
  },
  subheader: {
    fontSize: 14,
    color: colors.textSecondary,
    marginBottom: 20,
  },
  // --- Stats Row ---
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  statsCard: {
    flex: 1,
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
    marginHorizontal: 4,
    elevation: 2,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  statsNumber: {
    fontSize: 28,
    fontWeight: '900',
  },
  statsLabel: {
    fontSize: 11,
    color: colors.textDark,
    marginTop: 4,
    fontWeight: '600',
  },
  // --- Filter Pills ---
  filterContainer: {
    marginBottom: 12,
  },
  filterLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 8,
    fontWeight: '600',
  },
  filterPills: {
    flexDirection: 'row',
    gap: 8,
  },
  filterPill: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: colors.white,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  filterPillActive: {
    backgroundColor: colors.pinkLight,
    borderColor: colors.pinkLight,
  },
  filterPillText: {
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  filterPillTextActive: {
    color: colors.textDark,
  },
  // --- Sort Buttons ---
  sortContainer: {
    flexDirection: 'row',
    marginBottom: 20,
    gap: 10,
  },
  sortButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: colors.white,
  },
  sortButtonActive: {
    backgroundColor: colors.lightGreen,
  },
  sortButtonText: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  sortButtonTextActive: {
    color: colors.textDark,
    fontWeight: '600',
  },
  // --- Section Title ---
  sectionTitle: {
    fontSize: 20,
    fontWeight: "800",
    color: colors.textDark,
    marginBottom: 15,
  },
  // --- Alert Card ---
  alertCard: (riskColor) => ({
    backgroundColor: colors.white,
    borderLeftWidth: 6,
    borderLeftColor: riskColor,
    padding: 16,
    borderRadius: 12,
    marginBottom: 14,
    elevation: 3,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  }),
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  riskBadge: (riskColor) => ({
    backgroundColor: riskColor + '20',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  }),
  riskBadgeText: {
    fontSize: 11,
    fontWeight: '900',
    color: colors.textDark,
  },
  timestamp: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  categoryTag: {
    backgroundColor: colors.pinkLight,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    alignSelf: 'flex-start',
    marginBottom: 10,
  },
  categoryText: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.textDark,
  },
  alertTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.textDark,
    marginBottom: 4,
  },
  alertSubject: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.textSecondary,
    fontStyle: 'italic',
    marginBottom: 8,
  },
  alertPreview: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 18,
    marginBottom: 12,
  },
  residentInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#F5F5F5',
    padding: 10,
    borderRadius: 8,
    marginBottom: 12,
  },
  residentName: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textDark,
  },
  residentRoom: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  confidenceBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  confidenceLabel: {
    fontSize: 11,
    color: colors.textSecondary,
    marginRight: 8,
  },
  confidenceTrack: {
    flex: 1,
    height: 6,
    backgroundColor: '#E0E0E0',
    borderRadius: 3,
    marginRight: 8,
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 3,
  },
  confidenceValue: {
    fontSize: 12,
    fontWeight: '700',
  },
  tapHint: {
    fontSize: 12,
    color: colors.textSecondary,
    textAlign: 'right',
    fontStyle: 'italic',
  },
});