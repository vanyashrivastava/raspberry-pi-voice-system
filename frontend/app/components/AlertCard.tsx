import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import RiskBadge from "./RiskBadge";
import { colors } from "../../theme/colors";

export default function AlertCard({ item, onPress }) {
  return (
    <TouchableOpacity style={styles.card} onPress={onPress}>
      <View style={styles.header}>
        <Text style={styles.title}>{item.title || item.resident}</Text>
        <RiskBadge risk={item.risk} />
      </View>
      <Text style={styles.subtitle}>{item.subtitle || item.type}</Text>
      <Text style={styles.preview} numberOfLines={2}>
        {item.preview}
      </Text>
      <Text style={styles.timestamp}>{item.timestamp}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.alertCard,
    padding: 16,
    borderRadius: 14,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: "#FF6B6B",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
    flex: 1,
  },
  subtitle: {
    fontSize: 13,
    color: "#666",
    marginBottom: 6,
  },
  preview: {
    fontSize: 13,
    color: "#777",
    lineHeight: 18,
    marginBottom: 8,
  },
  timestamp: {
    fontSize: 11,
    color: "#999",
  },
});
