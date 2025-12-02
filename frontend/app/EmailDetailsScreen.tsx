import React, { useState } from "react";
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert, Animated } from "react-native";
import { colors } from "../theme/colors";

// Helper function to dynamically determine risk color
const getRiskColor = (risk) => {
  switch (risk?.toLowerCase()) {
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

export default function EmailDetailsScreen({ route, navigation }) {
  const { email } = route.params;
  const [action, setAction] = useState(null);
  const [expandedSection, setExpandedSection] = useState("content");

  const riskColor = getRiskColor(email.risk);

  // Suspicious keywords to highlight
  const suspiciousKeywords = [
    "verify", "confirm", "click here", "urgent", "immediately", "suspended",
    "locked", "expire", "act now", "limited time", "wire", "gift card",
    "social security", "medicare", "bank account", "password", "arrested",
    "grandson", "granddaughter", "refund", "winner", "congratulations"
  ];

  const highlightKeywords = (text) => {
    const words = text.split(" ");
    return words.map((word, index) => {
      const cleanWord = word.toLowerCase().replace(/[.,!?:]/g, "");
      const isSuspicious = suspiciousKeywords.some(keyword => 
        cleanWord.includes(keyword.toLowerCase())
      );
      return (
        <Text
          key={index}
          style={isSuspicious ? styles.highlightedKeyword : styles.normalText}
        >
          {word}{" "}
        </Text>
      );
    });
  };

  // Get scam-specific educational content
  const getScamEducation = (category) => {
    const education = {
      "Government Impersonation": {
        icon: "🏛️",
        title: "Government Impersonation Scam",
        description: "The government (SSA, IRS, Medicare) will NEVER email you asking for personal information, threaten to suspend benefits via email, or request immediate payment.",
        realAction: "Real government agencies send official mail and never demand immediate action.",
      },
      "Grandparent Scam": {
        icon: "👴",
        title: "Grandparent/Family Emergency Scam",
        description: "Scammers pretend to be grandchildren in emergency situations, requesting money via wire transfer or gift cards while asking victims to keep it secret.",
        realAction: "Always verify by calling the family member directly at their known phone number.",
      },
      "Healthcare Scam": {
        icon: "🏥",
        title: "Healthcare/Pharmacy Scam",
        description: "These scams impersonate pharmacies or healthcare providers to steal Medicare IDs, payment information, or sell fake medications.",
        realAction: "Contact your pharmacy or healthcare provider directly using official numbers.",
      },
      "Lottery/Prize Scam": {
        icon: "🎰",
        title: "Lottery/Prize Scam",
        description: "You cannot win a lottery or sweepstakes you didn't enter. Legitimate prizes never require upfront fees, especially via gift cards.",
        realAction: "Real prizes don't require payment. If you didn't enter, you didn't win.",
      },
      "Tech Support Scam": {
        icon: "💻",
        title: "Tech Support Scam",
        description: "Tech companies like Apple, Microsoft, and Google will never email you about security issues requiring immediate action through provided links.",
        realAction: "Go directly to the official website by typing the URL yourself.",
      },
      "Fake Order Scam": {
        icon: "📦",
        title: "Fake Order/Delivery Scam",
        description: "Scammers send fake order confirmations to create panic, hoping victims will click malicious links to 'cancel' orders they never placed.",
        realAction: "Check your actual account on the official website instead of clicking email links.",
      },
    };
    return education[category] || {
      icon: "⚠️",
      title: "Suspicious Email",
      description: "This email contains characteristics commonly found in scam messages.",
      realAction: "When in doubt, verify through official channels.",
    };
  };

  const scamInfo = getScamEducation(email.category);

  const handleAction = (actionType) => {
    setAction(actionType);
    
    const messages = {
      safe: {
        title: "Marked as Safe",
        message: `This email from "${email.title}" has been marked as safe. It will be removed from the alerts list.`
      },
      escalate: {
        title: "Escalated to Security",
        message: `This email has been escalated to the security team. ${email.resident}'s family will be notified.`
      },
      block: {
        title: "Sender Blocked",
        message: `The sender "${email.sender}" has been blocked. Future emails from this address will be automatically filtered.`
      },
      notify: {
        title: "Family Notified",
        message: `${email.resident}'s emergency contacts have been notified about this potential scam attempt.`
      }
    };

    Alert.alert(
      messages[actionType].title,
      messages[actionType].message,
      [{ text: "OK", onPress: () => actionType !== "notify" && actionType !== "block" && setTimeout(() => navigation.goBack(), 500) }]
    );
  };

  const CollapsibleSection = ({ title, icon, children, sectionKey }) => (
    <View style={styles.collapsibleSection}>
      <TouchableOpacity 
        style={styles.collapsibleHeader}
        onPress={() => setExpandedSection(expandedSection === sectionKey ? null : sectionKey)}
      >
        <Text style={styles.collapsibleTitle}>{icon} {title}</Text>
        <Text style={styles.expandIcon}>{expandedSection === sectionKey ? "▼" : "▶"}</Text>
      </TouchableOpacity>
      {expandedSection === sectionKey && (
        <View style={styles.collapsibleContent}>
          {children}
        </View>
      )}
    </View>
  );

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* HEADER */}
      <View style={styles.headerContainer}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← Back</Text>
        </TouchableOpacity>
        <View style={styles.riskBadge(riskColor)}>
          <Text style={styles.riskBadgeText}>{email.risk?.toUpperCase()} RISK</Text>
        </View>
      </View>

      {/* RESIDENT CARD */}
      <View style={styles.residentCard}>
        <View style={styles.residentAvatar}>
          <Text style={styles.avatarText}>{email.resident?.charAt(0) || "?"}</Text>
        </View>
        <View style={styles.residentDetails}>
          <Text style={styles.residentName}>{email.resident || "Unknown Resident"}</Text>
          <Text style={styles.residentRoom}>{email.room || "Room Unknown"}</Text>
        </View>
        <Text style={styles.timestamp}>{email.timestamp}</Text>
      </View>

      {/* SCAM CATEGORY BANNER */}
      <View style={[styles.categoryBanner, { backgroundColor: riskColor + '15' }]}>
        <Text style={styles.categoryIcon}>{scamInfo.icon}</Text>
        <View style={styles.categoryContent}>
          <Text style={[styles.categoryTitle, { color: riskColor }]}>{scamInfo.title}</Text>
          <Text style={styles.categoryDescription}>{scamInfo.description}</Text>
        </View>
      </View>

      {/* AI CONFIDENCE METER */}
      <View style={styles.aiSection}>
        <Text style={styles.aiTitle}>🤖 AI Detection Confidence</Text>
        <View style={styles.aiMeter}>
          <View style={styles.aiMeterTrack}>
            <View style={[styles.aiMeterFill, { width: `${email.aiConfidence || 85}%`, backgroundColor: riskColor }]} />
          </View>
          <Text style={[styles.aiConfidenceText, { color: riskColor }]}>{email.aiConfidence || 85}%</Text>
        </View>
        <Text style={styles.aiExplanation}>
          Our AI analyzed this email and is {email.aiConfidence || 85}% confident it's a scam attempt.
        </Text>
      </View>

      {/* EMAIL DETAILS - COLLAPSIBLE */}
      <CollapsibleSection title="Email Details" icon="📧" sectionKey="details">
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>From (Displayed):</Text>
          <Text style={styles.detailValue}>{email.sender}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Actual Sender:</Text>
          <Text style={[styles.detailValue, { color: colors.alertRed }]}>{email.senderReal || "Unknown"}</Text>
        </View>
        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Subject:</Text>
          <Text style={styles.detailValue}>{email.subject}</Text>
        </View>
      </CollapsibleSection>

      {/* EMAIL CONTENT - COLLAPSIBLE */}
      <CollapsibleSection title="Email Content (Highlighted)" icon="📄" sectionKey="content">
        <View style={styles.contentBox}>
          <Text style={styles.contentText}>
            {highlightKeywords(email.preview || "")}
          </Text>
        </View>
        <View style={styles.legendBox}>
          <Text style={styles.legendTitle}>Legend:</Text>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: '#FFE4E1' }]} />
            <Text style={styles.legendText}>Highlighted = Suspicious keyword</Text>
          </View>
        </View>
      </CollapsibleSection>

      {/* RED FLAGS - COLLAPSIBLE */}
      <CollapsibleSection title="Red Flags Detected" icon="🚩" sectionKey="flags">
        <View style={styles.flagsContainer}>
          {(email.flags || ["Suspicious content detected"]).map((flag, index) => (
            <View key={index} style={styles.flagItem}>
              <Text style={styles.flagIcon}>⚠️</Text>
              <Text style={styles.flagText}>{flag}</Text>
            </View>
          ))}
        </View>
      </CollapsibleSection>

      {/* WHAT TO TELL RESIDENT */}
      <View style={styles.talkingPointsCard}>
        <Text style={styles.talkingPointsTitle}>💬 Talking Points for Staff</Text>
        <Text style={styles.talkingPointsSubtitle}>When discussing with {email.resident?.split(" ")[0] || "the resident"}:</Text>
        <View style={styles.talkingPoint}>
          <Text style={styles.talkingPointBullet}>1.</Text>
          <Text style={styles.talkingPointText}>
            "This email was flagged by our safety system because it shows signs of a common scam."
          </Text>
        </View>
        <View style={styles.talkingPoint}>
          <Text style={styles.talkingPointBullet}>2.</Text>
          <Text style={styles.talkingPointText}>
            "{scamInfo.realAction}"
          </Text>
        </View>
        <View style={styles.talkingPoint}>
          <Text style={styles.talkingPointBullet}>3.</Text>
          <Text style={styles.talkingPointText}>
            "You did nothing wrong - these scammers are very sophisticated. We're here to help protect you."
          </Text>
        </View>
      </View>

      {/* SUGGESTED ACTIONS */}
      <View style={styles.suggestionsCard}>
        <Text style={styles.suggestionsTitle}>✅ Recommended Actions</Text>
        <View style={styles.suggestionItem}>
          <Text style={styles.checkmark}>•</Text>
          <Text style={styles.suggestionText}>Do NOT click any links in the email</Text>
        </View>
        <View style={styles.suggestionItem}>
          <Text style={styles.checkmark}>•</Text>
          <Text style={styles.suggestionText}>Do NOT reply or provide any information</Text>
        </View>
        <View style={styles.suggestionItem}>
          <Text style={styles.checkmark}>•</Text>
          <Text style={styles.suggestionText}>Contact {email.category?.includes("Medicare") ? "Medicare at 1-800-MEDICARE" : "the real organization"} directly</Text>
        </View>
        <View style={styles.suggestionItem}>
          <Text style={styles.checkmark}>•</Text>
          <Text style={styles.suggestionText}>Document this incident in resident's file</Text>
        </View>
      </View>

      {/* ACTION BUTTONS */}
      {!action && (
        <View style={styles.actionSection}>
          <Text style={styles.actionSectionTitle}>Take Action</Text>
          
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={styles.safeButton}
              onPress={() => handleAction("safe")}
            >
              <Text style={styles.buttonIcon}>✓</Text>
              <Text style={styles.safeButtonText}>Mark Safe</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.escalateButton}
              onPress={() => handleAction("escalate")}
            >
              <Text style={styles.buttonIcon}>🚨</Text>
              <Text style={styles.escalateButtonText}>Escalate</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={styles.blockButton}
              onPress={() => handleAction("block")}
            >
              <Text style={styles.buttonIcon}>🚫</Text>
              <Text style={styles.blockButtonText}>Block Sender</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.notifyButton}
              onPress={() => handleAction("notify")}
            >
              <Text style={styles.buttonIcon}>👨‍👩‍👧</Text>
              <Text style={styles.notifyButtonText}>Notify Family</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* CONFIRMATION */}
      {action && (
        <View style={[styles.confirmBox, { backgroundColor: action === "safe" ? colors.lightGreen : colors.pinkLight }]}>
          <Text style={styles.confirmText}>
            {action === "safe" && "✓ Marked as safe"}
            {action === "escalate" && "🚨 Escalated to security team"}
            {action === "block" && "🚫 Sender has been blocked"}
            {action === "notify" && "👨‍👩‍👧 Family has been notified"}
          </Text>
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
    paddingTop: 50,
    paddingHorizontal: 20,
  },
  headerContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  backButton: {
    fontSize: 16,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  riskBadge: (riskColor) => ({
    backgroundColor: riskColor,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
  }),
  riskBadgeText: {
    fontSize: 12,
    fontWeight: '900',
    color: colors.white,
  },
  // --- Resident Card ---
  residentCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.white,
    padding: 16,
    borderRadius: 16,
    marginBottom: 16,
    elevation: 2,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  residentAvatar: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: colors.pinkLight,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 14,
  },
  avatarText: {
    fontSize: 24,
    fontWeight: '800',
    color: colors.textDark,
  },
  residentDetails: {
    flex: 1,
  },
  residentName: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.textDark,
  },
  residentRoom: {
    fontSize: 14,
    color: colors.textSecondary,
  },
  timestamp: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  // --- Category Banner ---
  categoryBanner: {
    flexDirection: 'row',
    padding: 16,
    borderRadius: 16,
    marginBottom: 16,
    alignItems: 'flex-start',
  },
  categoryIcon: {
    fontSize: 32,
    marginRight: 12,
  },
  categoryContent: {
    flex: 1,
  },
  categoryTitle: {
    fontSize: 16,
    fontWeight: '800',
    marginBottom: 6,
  },
  categoryDescription: {
    fontSize: 13,
    color: colors.textDark,
    lineHeight: 18,
  },
  // --- AI Section ---
  aiSection: {
    backgroundColor: colors.white,
    padding: 16,
    borderRadius: 16,
    marginBottom: 16,
    elevation: 2,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  aiTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: 12,
  },
  aiMeter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  aiMeterTrack: {
    flex: 1,
    height: 10,
    backgroundColor: '#E0E0E0',
    borderRadius: 5,
    marginRight: 12,
  },
  aiMeterFill: {
    height: '100%',
    borderRadius: 5,
  },
  aiConfidenceText: {
    fontSize: 18,
    fontWeight: '900',
  },
  aiExplanation: {
    fontSize: 12,
    color: colors.textSecondary,
    fontStyle: 'italic',
  },
  // --- Collapsible Sections ---
  collapsibleSection: {
    backgroundColor: colors.white,
    borderRadius: 16,
    marginBottom: 12,
    overflow: 'hidden',
    elevation: 2,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  collapsibleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  collapsibleTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textDark,
  },
  expandIcon: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  collapsibleContent: {
    paddingHorizontal: 16,
    paddingBottom: 16,
  },
  // --- Detail Rows ---
  detailRow: {
    marginBottom: 12,
  },
  detailLabel: {
    fontSize: 11,
    color: colors.textSecondary,
    fontWeight: '600',
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  detailValue: {
    fontSize: 14,
    color: colors.textDark,
    backgroundColor: '#F5F5F5',
    padding: 10,
    borderRadius: 8,
  },
  // --- Content Box ---
  contentBox: {
    backgroundColor: '#FFFAF0',
    padding: 14,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: colors.alertOrange,
  },
  contentText: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  normalText: {
    fontSize: 14,
    color: colors.textDark,
    lineHeight: 22,
  },
  highlightedKeyword: {
    fontSize: 14,
    color: colors.alertRed,
    fontWeight: 'bold',
    backgroundColor: '#FFE4E1',
    lineHeight: 22,
  },
  legendBox: {
    marginTop: 12,
    padding: 10,
    backgroundColor: '#F5F5F5',
    borderRadius: 8,
  },
  legendTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.textSecondary,
    marginBottom: 6,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 2,
    marginRight: 8,
  },
  legendText: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  // --- Flags Container ---
  flagsContainer: {
    gap: 8,
  },
  flagItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF5F5',
    padding: 12,
    borderRadius: 10,
  },
  flagIcon: {
    fontSize: 16,
    marginRight: 10,
  },
  flagText: {
    fontSize: 14,
    color: colors.textDark,
    fontWeight: '600',
  },
  // --- Talking Points ---
  talkingPointsCard: {
    backgroundColor: colors.lightGreen,
    padding: 16,
    borderRadius: 16,
    marginBottom: 12,
  },
  talkingPointsTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textDark,
    marginBottom: 4,
  },
  talkingPointsSubtitle: {
    fontSize: 13,
    color: colors.textDark,
    opacity: 0.7,
    marginBottom: 12,
  },
  talkingPoint: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  talkingPointBullet: {
    fontSize: 14,
    fontWeight: '800',
    color: colors.textDark,
    marginRight: 8,
  },
  talkingPointText: {
    flex: 1,
    fontSize: 14,
    color: colors.textDark,
    lineHeight: 20,
  },
  // --- Suggestions ---
  suggestionsCard: {
    backgroundColor: '#F0F8FF',
    padding: 16,
    borderRadius: 16,
    marginBottom: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#4A90D9',
  },
  suggestionsTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textDark,
    marginBottom: 12,
  },
  suggestionItem: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  checkmark: {
    fontSize: 16,
    color: '#4A90D9',
    marginRight: 10,
    fontWeight: '800',
  },
  suggestionText: {
    flex: 1,
    fontSize: 14,
    color: colors.textDark,
    lineHeight: 20,
  },
  // --- Action Buttons ---
  actionSection: {
    marginTop: 8,
  },
  actionSectionTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.textDark,
    marginBottom: 12,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  safeButton: {
    flex: 1,
    backgroundColor: colors.lightGreen,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  safeButtonText: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textDark,
  },
  escalateButton: {
    flex: 1,
    backgroundColor: colors.alertRed,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  escalateButtonText: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.white,
  },
  blockButton: {
    flex: 1,
    backgroundColor: colors.pinkLight,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  blockButtonText: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textDark,
  },
  notifyButton: {
    flex: 1,
    backgroundColor: '#E8F4FD',
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  notifyButtonText: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textDark,
  },
  buttonIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  // --- Confirmation ---
  confirmBox: {
    padding: 18,
    borderRadius: 14,
    alignItems: 'center',
  },
  confirmText: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textDark,
  },
});
