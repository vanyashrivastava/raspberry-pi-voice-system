import React, { useState } from "react";
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from "react-native";
import { colors } from "../theme/colors";
import RiskBadge from "./components/RiskBadge";

export default function EmailDetailsScreen({ route, navigation }) {
  const { email } = route.params;
  const [action, setAction] = useState(null);

  const suspiciousKeywords = ["verify", "confirm", "click here", "unusual", "locked"];

  const highlightKeywords = (text) => {
    const words = text.split(" ");
    return words.map((word, index) => {
      const cleanWord = word.toLowerCase().replace(/[.,!?]/g, "");
      const isSuspicious = suspiciousKeywords.includes(cleanWord);
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

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>{email.title}</Text>
        <RiskBadge risk={email.risk} />
      </View>

      <View style={styles.section}>
        <Text style={styles.label}>From:</Text>
        <Text style={styles.value}>{email.sender}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.label}>Subject:</Text>
        <Text style={styles.value}>{email.subject}</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.label}>Email Content:</Text>
        <View style={styles.contentBox}>
          <Text style={styles.content}>
            {highlightKeywords(
              email.preview + " This is a suspicious email attempting to collect personal information. Do not click any links or provide account details."
            )}
          </Text>
        </View>
      </View>

      <View style={styles.warningBox}>
        <Text style={styles.warningTitle}>⚠️ Suspicious Indicators Found:</Text>
        <Text style={styles.warningText}>• Requests personal or financial information</Text>
        <Text style={styles.warningText}>• Urges immediate action</Text>
        <Text style={styles.warningText}>• Generic greeting</Text>
      </View>

      <View style={styles.suggestionsBox}>
        <Text style={styles.suggestionsTitle}>Suggested Actions:</Text>
        <Text style={styles.suggestionText}>
          • Do not click any links in the email{"\n"}
          • Do not provide account information{"\n"}
          • Contact the institution directly using official number{"\n"}
          • Report to IT department
        </Text>
      </View>

      {!action && (
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.safeButton}
            onPress={() => {
              setAction("safe");
              setTimeout(() => navigation.goBack(), 500);
            }}
          >
            <Text style={styles.safeButtonText}>Mark Safe</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.escalateButton}
            onPress={() => {
              setAction("escalate");
              setTimeout(() => navigation.goBack(), 500);
            }}
          >
            <Text style={styles.escalateButtonText}>Escalate</Text>
          </TouchableOpacity>
        </View>
      )}

      {action && (
        <View style={styles.confirmBox}>
          <Text style={styles.confirmText}>
            {action === "safe" ? "✓ Marked as safe" : "✓ Escalated to security team"}
          </Text>
        </View>
      )}
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
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: colors.textDark,
    flex: 1,
  },
  section: {
    marginBottom: 16,
  },
  label: {
    fontSize: 12,
    color: "#999",
    marginBottom: 4,
    fontWeight: "bold",
  },
  value: {
    fontSize: 14,
    color: colors.textDark,
    backgroundColor: "#f5f5f5",
    padding: 12,
    borderRadius: 8,
  },
  contentBox: {
    backgroundColor: "#fafafa",
    padding: 14,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: colors.pink,
  },
  content: {
    fontSize: 13,
    color: "#555",
    lineHeight: 20,
  },
  normalText: {
    fontSize: 13,
    color: "#555",
  },
  highlightedKeyword: {
    fontSize: 13,
    color: "#FF6B6B",
    fontWeight: "bold",
    backgroundColor: "#FFE4E1",
  },
  warningBox: {
    backgroundColor: "#FFF5F5",
    borderLeftWidth: 4,
    borderLeftColor: "#FF6B6B",
    padding: 14,
    borderRadius: 8,
    marginVertical: 16,
  },
  warningTitle: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#FF6B6B",
    marginBottom: 8,
  },
  warningText: {
    fontSize: 13,
    color: "#D9534F",
    marginBottom: 4,
  },
  suggestionsBox: {
    backgroundColor: "#F0F8FF",
    borderLeftWidth: 4,
    borderLeftColor: colors.green,
    padding: 14,
    borderRadius: 8,
    marginVertical: 16,
  },
  suggestionsTitle: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#51CF66",
    marginBottom: 8,
  },
  suggestionText: {
    fontSize: 13,
    color: "#2B7A2B",
    lineHeight: 20,
  },
  buttonContainer: {
    flexDirection: "row",
    gap: 12,
    marginTop: 20,
  },
  safeButton: {
    flex: 1,
    backgroundColor: colors.green,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  safeButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
  },
  escalateButton: {
    flex: 1,
    backgroundColor: "#FF6B6B",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  escalateButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#fff",
  },
  confirmBox: {
    backgroundColor: colors.green,
    padding: 16,
    borderRadius: 10,
    marginTop: 20,
    alignItems: "center",
  },
  confirmText: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
  },
});
