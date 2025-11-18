import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  TouchableOpacity,
} from "react-native";
import { colors } from "../theme/colors";
import RiskBadge from "./components/RiskBadge";

export default function ResidentProfileScreen({ route }) {
  const { resident } = route.params;
  const [notes, setNotes] = useState("");

  const mockAlerts = [
    "Email phishing attempt - 2 days ago",
    "Suspicious phone call - 5 days ago",
    "Unusual banking activity - 1 week ago",
  ];

  return (
    <ScrollView style={styles.container}>
      <View style={styles.profileHeader}>
        <View style={styles.avatar}>
          <Text style={styles.initials}>
            {resident.name
              .split(" ")
              .map((n) => n[0])
              .join("")}
          </Text>
        </View>
        <Text style={styles.name}>{resident.name}</Text>
        <Text style={styles.age}>Age {resident.age}</Text>
        <RiskBadge risk={resident.risk} />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Risk Assessment</Text>
        <View style={styles.riskBox}>
          <Text style={styles.riskLabel}>Current Risk Level</Text>
          <Text style={styles.riskValue}>
            {resident.risk === "High" ? "⚠️" : "✓"} {resident.risk}
          </Text>
          <Text style={styles.riskDescription}>
            {resident.risk === "High"
              ? "Increased monitoring recommended"
              : "Low vulnerability profile"}
          </Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Recent Scam Alerts</Text>
        {mockAlerts.map((alert, index) => (
          <View key={index} style={styles.alertItem}>
            <Text style={styles.alertBullet}>•</Text>
            <Text style={styles.alertText}>{alert}</Text>
          </View>
        ))}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Emergency Contact</Text>
        <View style={styles.contactBox}>
          <View style={styles.contactField}>
            <Text style={styles.contactLabel}>Primary Contact</Text>
            <Text style={styles.contactValue}>
              {resident.name.split(" ")[1] || "Family"} Johnson
            </Text>
          </View>
          <View style={styles.contactField}>
            <Text style={styles.contactLabel}>Phone</Text>
            <Text style={styles.contactValue}>(555) 123-4567</Text>
          </View>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Suggested Steps</Text>
        <View style={styles.suggestionItem}>
          <Text style={styles.suggestionBullet}>✓</Text>
          <Text style={styles.suggestionText}>
            Increase call monitoring frequency
          </Text>
        </View>
        <View style={styles.suggestionItem}>
          <Text style={styles.suggestionBullet}>✓</Text>
          <Text style={styles.suggestionText}>
            Brief resident on common scam tactics
          </Text>
        </View>
        <View style={styles.suggestionItem}>
          <Text style={styles.suggestionBullet}>✓</Text>
          <Text style={styles.suggestionText}>
            Review banking and email security
          </Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notes</Text>
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
          <Text style={styles.saveButtonText}>Save Notes</Text>
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
  profileHeader: {
    alignItems: "center",
    marginBottom: 30,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.pink,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 12,
  },
  initials: {
    fontSize: 32,
    fontWeight: "bold",
    color: colors.textDark,
  },
  name: {
    fontSize: 22,
    fontWeight: "bold",
    color: colors.textDark,
    marginBottom: 4,
  },
  age: {
    fontSize: 14,
    color: "#666",
    marginBottom: 12,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
    marginBottom: 12,
  },
  riskBox: {
    backgroundColor: colors.alertCard,
    padding: 16,
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: "#FF6B6B",
  },
  riskLabel: {
    fontSize: 12,
    color: "#999",
    marginBottom: 8,
  },
  riskValue: {
    fontSize: 20,
    fontWeight: "bold",
    color: colors.textDark,
    marginBottom: 8,
  },
  riskDescription: {
    fontSize: 13,
    color: "#666",
  },
  alertItem: {
    flexDirection: "row",
    marginBottom: 12,
    paddingLeft: 8,
  },
  alertBullet: {
    fontSize: 16,
    color: "#FF6B6B",
    marginRight: 8,
  },
  alertText: {
    fontSize: 13,
    color: "#555",
    flex: 1,
  },
  contactBox: {
    backgroundColor: colors.alertCard,
    borderRadius: 10,
    padding: 16,
  },
  contactField: {
    marginBottom: 12,
  },
  contactLabel: {
    fontSize: 12,
    color: "#999",
    marginBottom: 4,
  },
  contactValue: {
    fontSize: 14,
    color: colors.textDark,
    fontWeight: "500",
  },
  suggestionItem: {
    flexDirection: "row",
    marginBottom: 12,
    paddingLeft: 8,
  },
  suggestionBullet: {
    fontSize: 16,
    color: colors.green,
    marginRight: 8,
  },
  suggestionText: {
    fontSize: 13,
    color: "#555",
    flex: 1,
  },
  notesInput: {
    backgroundColor: colors.alertCard,
    borderRadius: 10,
    padding: 12,
    fontSize: 13,
    color: colors.textDark,
    textAlignVertical: "top",
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#f0f0f0",
  },
  saveButton: {
    backgroundColor: colors.green,
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
  },
});
