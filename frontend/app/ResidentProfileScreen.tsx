import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Dimensions,
} from "react-native";
import { colors } from "../theme/colors";
import RiskBadge from "./components/RiskBadge";

const { width } = Dimensions.get("window");

export default function ResidentProfileScreen({ route, navigation }) {
  const { resident } = route.params;
  const [notes, setNotes] = useState("");
  const [expandedSections, setExpandedSections] = useState({
    alerts: true,
    contact: false,
    suggestions: false,
  });

  // All available scam alerts to choose from
  const allMockAlerts = [
    { text: "Email phishing attempt", date: "2 days ago", severity: "High" },
    { text: "Suspicious phone call", date: "5 days ago", severity: "Medium" },
    { text: "Unusual banking activity", date: "1 week ago", severity: "High" },
    { text: "Fraudulent wire transfer request", date: "1 week ago", severity: "High" },
    { text: "IRS impersonation call", date: "2 weeks ago", severity: "High" },
    { text: "Tech support scam attempt", date: "3 weeks ago", severity: "Medium" },
  ];

  // Get only the alerts that match the resident's alert count
  const mockAlerts = allMockAlerts.slice(0, resident.alerts);

  // Calculate birth year from age (approximation)
  const currentYear = new Date().getFullYear();
  const birthYear = currentYear - resident.age;

  // Get initials for avatar
  const getInitials = (name) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  // Get risk color
  const getRiskColor = (risk) => {
    switch (risk) {
      case "High":
        return "#FF6B6B";
      case "Medium":
        return "#FFA500";
      case "Low":
        return "#2ECC71";
      default:
        return colors.green;
    }
  };

  // Get background color based on severity
  const getSeverityBackground = (severity) => {
    switch (severity) {
      case "High":
        return "#FFE5E5";
      case "Medium":
        return "#FFF8E5";
      case "Low":
        return "#E5F5F0";
      default:
        return "#F5F5F5";
    }
  };

  // Toggle section expansion
  const toggleSection = (section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  return (
    <ScrollView style={styles.container}>
      {/* Back Button */}
      <TouchableOpacity
        style={styles.backButton}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.backButtonText}>← Back</Text>
      </TouchableOpacity>

      {/* Enhanced Profile Header */}
      <View style={styles.profileHeader}>
        <View style={[styles.avatar, { backgroundColor: colors.pink }]}>
          <Text style={styles.initials}>{getInitials(resident.name)}</Text>
        </View>
        <Text style={styles.name}>{resident.name}</Text>
        <View style={styles.bioRow}>
          <Text style={styles.bioText}>👤 Age {resident.age}</Text>
          <Text style={styles.bioDot}>•</Text>
          <Text style={styles.bioText}>🎂 {resident.birthMonth}/{resident.birthDay}/{birthYear}</Text>
        </View>
        
        {/* Risk Status Card */}
        <View style={[styles.riskStatusCard, { borderLeftColor: getRiskColor(resident.risk) }]}>
          <View style={styles.riskStatusContent}>
            <Text style={styles.riskStatusLabel}>Risk Level</Text>
            <RiskBadge risk={resident.risk} />
          </View>
          <Text style={styles.alertCountBadge}>🔔 {resident.alerts}</Text>
        </View>
      </View>

      {/* Risk Assessment Section */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>📊 Risk Assessment</Text>
        </View>
        <View style={[styles.riskBox, { borderLeftColor: getRiskColor(resident.risk) }]}>
          <View style={styles.riskBoxRow}>
            <Text style={styles.riskLabel}>Current Risk Level</Text>
            <Text style={[styles.riskValue, { color: getRiskColor(resident.risk) }]}>
              {resident.risk === "High" ? "⚠️" : resident.risk === "Medium" ? "⚡" : "✓"} {resident.risk}
            </Text>
          </View>
          <Text style={styles.riskDescription}>
            {resident.risk === "High"
              ? "Increased monitoring recommended. Consider more frequent check-ins."
              : resident.risk === "Medium"
              ? "Moderate vigilance advised. Regular monitoring encouraged."
              : "Low vulnerability profile. Standard monitoring sufficient."}
          </Text>
        </View>
      </View>

      {/* Recent Scam Alerts Section */}
      <View style={styles.section}>
        <TouchableOpacity
          style={styles.sectionHeader}
          onPress={() => toggleSection("alerts")}
        >
          <Text style={styles.sectionTitle}>🚨 Recent Scam Alerts ({mockAlerts.length})</Text>
          <Text style={styles.expandIcon}>{expandedSections.alerts ? "▼" : "▶"}</Text>
        </TouchableOpacity>
        
        {expandedSections.alerts && (
          <View>
            {mockAlerts.map((alert, index) => (
              <View
                key={index}
                style={[
                  styles.alertCard,
                  { backgroundColor: getSeverityBackground(alert.severity) },
                ]}
              >
                <View style={styles.alertHeader}>
                  <View style={styles.alertTitleRow}>
                    <Text
                      style={[
                        styles.alertSeverity,
                        { color: getRiskColor(alert.severity) },
                      ]}
                    >
                      {alert.severity === "High" ? "●●●" : alert.severity === "Medium" ? "●●" : "●"}
                    </Text>
                    <Text style={styles.alertText}>{alert.text}</Text>
                  </View>
                </View>
                <Text style={styles.alertDate}>{alert.date}</Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Emergency Contact Section */}
      <View style={styles.section}>
        <TouchableOpacity
          style={styles.sectionHeader}
          onPress={() => toggleSection("contact")}
        >
          <Text style={styles.sectionTitle}>📞 Emergency Contact</Text>
          <Text style={styles.expandIcon}>{expandedSections.contact ? "▼" : "▶"}</Text>
        </TouchableOpacity>

        {expandedSections.contact && (
          <View style={styles.contactBox}>
            <View style={styles.contactField}>
              <Text style={styles.contactLabel}>Primary Contact</Text>
              <Text style={styles.contactValue}>
                {resident.name.split(" ")[1] || "Family"} Johnson
              </Text>
            </View>
            <View style={styles.contactDivider} />
            <View style={styles.contactField}>
              <Text style={styles.contactLabel}>Phone Number</Text>
              <Text style={styles.contactValue}>(555) 123-4567</Text>
            </View>
            <View style={styles.contactDivider} />
            <View style={styles.contactField}>
              <Text style={styles.contactLabel}>Email</Text>
              <Text style={styles.contactValue}>contact@example.com</Text>
            </View>
          </View>
        )}
      </View>

      {/* Suggested Steps Section */}
      <View style={styles.section}>
        <TouchableOpacity
          style={styles.sectionHeader}
          onPress={() => toggleSection("suggestions")}
        >
          <Text style={styles.sectionTitle}>✓ Protective Measures</Text>
          <Text style={styles.expandIcon}>{expandedSections.suggestions ? "▼" : "▶"}</Text>
        </TouchableOpacity>

        {expandedSections.suggestions && (
          <View>
            <View style={styles.suggestionItem}>
              <View style={styles.suggestionCheckbox} />
              <Text style={styles.suggestionText}>
                Increase call monitoring frequency
              </Text>
            </View>
            <View style={styles.suggestionItem}>
              <View style={styles.suggestionCheckbox} />
              <Text style={styles.suggestionText}>
                Brief resident on common scam tactics
              </Text>
            </View>
            <View style={styles.suggestionItem}>
              <View style={styles.suggestionCheckbox} />
              <Text style={styles.suggestionText}>
                Review banking and email security
              </Text>
            </View>
            <View style={styles.suggestionItem}>
              <View style={styles.suggestionCheckbox} />
              <Text style={styles.suggestionText}>
                Set up two-factor authentication
              </Text>
            </View>
          </View>
        )}
      </View>

      {/* Notes Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📝 Personal Notes</Text>
        <TextInput
          style={styles.notesInput}
          placeholder="Add personal notes or observations..."
          placeholderTextColor="#aaa"
          multiline
          numberOfLines={4}
          value={notes}
          onChangeText={setNotes}
        />
        <TouchableOpacity style={styles.saveButton}>
          <Text style={styles.saveButtonText}>💾 Save Notes</Text>
        </TouchableOpacity>
      </View>

      {/* Quick Action Buttons */}
      <View style={styles.actionButtonsContainer}>
        <TouchableOpacity style={styles.actionButton}>
          <Text style={styles.actionButtonText}>📞 Call Contact</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionButton, styles.actionButtonSecondary]}>
          <Text style={styles.actionButtonTextSecondary}>📧 Send Alert</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.white,
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  backButton: {
    paddingVertical: 10,
    paddingHorizontal: 12,
    marginBottom: 16,
    backgroundColor: "#F5F5F5",
    borderRadius: 10,
    alignSelf: "flex-start",
    borderWidth: 1,
    borderColor: "#E0E0E0",
  },
  backButtonText: {
    fontSize: 15,
    fontWeight: "600",
    color: colors.textDark,
  },
  profileHeader: {
    alignItems: "center",
    marginBottom: 24,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  avatar: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: colors.pink,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 5,
  },
  initials: {
    fontSize: 40,
    fontWeight: "bold",
    color: colors.textDark,
  },
  name: {
    fontSize: 26,
    fontWeight: "bold",
    color: colors.textDark,
    marginBottom: 4,
  },
  age: {
    fontSize: 15,
    color: colors.textSecondary,
    marginBottom: 14,
    fontWeight: "500",
  },
  bioRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
    gap: 8,
  },
  bioText: {
    fontSize: 15,
    color: colors.textSecondary,
    fontWeight: "500",
  },
  bioDot: {
    fontSize: 14,
    color: colors.textSecondary,
  },
  riskStatusCard: {
    backgroundColor: "#F9F9F9",
    borderLeftWidth: 5,
    borderRadius: 12,
    padding: 14,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 12,
    width: "100%",
  },
  riskStatusContent: {
    flex: 1,
  },
  riskStatusLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 6,
  },
  alertCountBadge: {
    fontSize: 16,
    fontWeight: "700",
    color: "#996600",
    backgroundColor: "#FFF3CD",
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 8,
  },
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingBottom: 12,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: colors.textDark,
  },
  expandIcon: {
    fontSize: 12,
    color: colors.textSecondary,
    fontWeight: "600",
  },
  riskBox: {
    backgroundColor: "#F9F9F9",
    borderLeftWidth: 5,
    padding: 16,
    borderRadius: 12,
  },
  riskBoxRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  riskLabel: {
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: "500",
  },
  riskValue: {
    fontSize: 20,
    fontWeight: "bold",
    color: colors.textDark,
  },
  riskDescription: {
    fontSize: 13,
    color: "#666",
    lineHeight: 18,
  },
  alertCard: {
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 4,
    borderLeftColor: "#FF6B6B",
  },
  alertHeader: {
    marginBottom: 8,
  },
  alertTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  alertSeverity: {
    fontSize: 12,
    fontWeight: "bold",
    letterSpacing: 1,
  },
  alertText: {
    fontSize: 14,
    fontWeight: "600",
    color: colors.textDark,
    flex: 1,
  },
  alertDate: {
    fontSize: 12,
    color: colors.textSecondary,
    fontWeight: "500",
  },
  contactBox: {
    backgroundColor: "#F9F9F9",
    borderRadius: 12,
    padding: 16,
  },
  contactField: {
    marginBottom: 12,
  },
  contactLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 6,
    fontWeight: "500",
  },
  contactValue: {
    fontSize: 15,
    color: colors.textDark,
    fontWeight: "600",
  },
  contactDivider: {
    height: 1,
    backgroundColor: "#E0E0E0",
    marginVertical: 12,
  },
  suggestionItem: {
    flexDirection: "row",
    marginBottom: 14,
    alignItems: "center",
  },
  suggestionCheckbox: {
    width: 20,
    height: 20,
    borderRadius: 4,
    backgroundColor: colors.green,
    marginRight: 12,
    justifyContent: "center",
    alignItems: "center",
  },
  suggestionText: {
    fontSize: 14,
    color: "#555",
    flex: 1,
    fontWeight: "500",
  },
  notesInput: {
    backgroundColor: "#F9F9F9",
    borderRadius: 12,
    padding: 14,
    fontSize: 14,
    color: colors.textDark,
    textAlignVertical: "top",
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#E0E0E0",
  },
  saveButton: {
    backgroundColor: colors.green,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 16,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: "700",
    color: colors.textDark,
  },
  actionButtonsContainer: {
    flexDirection: "row",
    gap: 12,
    marginTop: 12,
  },
  actionButton: {
    flex: 1,
    backgroundColor: colors.pink,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  actionButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: colors.textDark,
  },
  actionButtonSecondary: {
    backgroundColor: colors.green,
  },
  actionButtonTextSecondary: {
    fontSize: 15,
    fontWeight: "700",
    color: colors.textDark,
  },
});
