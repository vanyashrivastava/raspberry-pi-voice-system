import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { colors } from "../theme/colors";

export default function AlertDetailsScreen({ route }) {
  const { alert } = route.params;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Alert Details</Text>

      <View style={styles.card}>
        <Text style={styles.label}>Resident:</Text>
        <Text style={styles.value}>{alert.resident}</Text>

        <Text style={styles.label}>Type:</Text>
        <Text style={styles.value}>{alert.type}</Text>

        <Text style={styles.label}>Risk Level:</Text>
        <Text style={styles.value}>{alert.risk}</Text>
      </View>

      <Text style={styles.suggestion}>
        Suggested Action:  
        {"\n"}
        • Verify the resident did NOT share personal or financial info.{"\n"}
        • Contact their family if necessary.{"\n"}
        • Report suspicious numbers or messages to the FTC.{"\n"}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.pink,
    padding: 30,
    paddingTop: 60,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 20,
  },
  card: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 14,
    marginBottom: 20,
  },
  label: {
    fontSize: 14,
    color: "#777",
    marginTop: 8,
  },
  value: {
    fontSize: 18,
    fontWeight: "bold",
    color: colors.textDark,
  },
  suggestion: {
    fontSize: 16,
    color: colors.textDark,
    lineHeight: 24,
  },
});
